import time
import torch
import soundfile as sf
import gc
import os

wav_path = os.path.join(os.path.dirname(__file__), "..", "..", "120报警电话16k.wav")

def extract_segments(speech_probs, threshold=0.5, sampling_rate=16000, window_size_samples=512):
    # Standard Silero logic implementation for baseline comparison
    min_speech_duration_ms = 250
    max_speech_duration_s = float('inf')
    min_silence_duration_ms = 100
    speech_pad_ms = 30
    min_silence_at_max_speech = 98
    use_max_poss_sil_at_max_speech = True

    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = sampling_rate * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * min_silence_at_max_speech / 1000

    audio_length_samples = len(speech_probs) * window_size_samples
    triggered = False
    speeches = []
    current_speech = {}
    neg_threshold = max(threshold - 0.15, 0.01)
    temp_end = 0
    prev_end = next_start = 0
    possible_ends = []

    for i, speech_prob in enumerate(speech_probs):
        cur_sample = window_size_samples * i

        if (speech_prob >= threshold) and temp_end:
            sil_dur = cur_sample - temp_end
            if sil_dur > min_silence_samples_at_max_speech:
                possible_ends.append((temp_end, sil_dur))
            temp_end = 0
            if next_start < prev_end:
                next_start = cur_sample

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech['start'] = cur_sample
            continue

        if triggered and (cur_sample - current_speech['start'] > max_speech_samples):
            if use_max_poss_sil_at_max_speech and possible_ends:
                prev_end, dur = max(possible_ends, key=lambda x: x[1])
                current_speech['end'] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                next_start = prev_end + dur

                if next_start < prev_end + cur_sample:
                    current_speech['start'] = next_start
                else:
                    triggered = False
                prev_end = next_start = temp_end = 0
                possible_ends = []
            else:
                if prev_end:
                    current_speech['end'] = prev_end
                    speeches.append(current_speech)
                    current_speech = {}
                    if next_start < prev_end:
                        triggered = False
                    else:
                        current_speech['start'] = next_start
                    prev_end = next_start = temp_end = 0
                    possible_ends = []
                else:
                    current_speech['end'] = cur_sample
                    speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    possible_ends = []
                    continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = cur_sample
            sil_dur_now = cur_sample - temp_end

            if not use_max_poss_sil_at_max_speech and sil_dur_now > min_silence_samples_at_max_speech:
                prev_end = temp_end

            if sil_dur_now < min_silence_samples:
                continue
            else:
                current_speech['end'] = temp_end
                if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                possible_ends = []
                continue

    if current_speech and (audio_length_samples - current_speech['start']) > min_speech_samples:
        current_speech['end'] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech['start'] = int(max(0, speech['start'] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i+1]['start'] - speech['end']
            if silence_duration < 2 * speech_pad_samples:
                speech['end'] += int(silence_duration // 2)
                speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - silence_duration // 2))
            else:
                speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))
                speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - speech_pad_samples))
        else:
            speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))

    return speeches

def simulate_dynamic_batching(audio, model, concurrency, max_batch_size, sampling_rate=16000):
    window_size_samples = 512
    audio_length_samples = len(audio)
    
    # 外部状态管理，模拟给 N 个并发流维护 state
    global_context = torch.zeros(concurrency, 64)
    global_state = torch.zeros(2, concurrency, 128)
    
    raw_model = model._model # 获取底层 JIT RNN 模型
    
    # 我们只记录第一个连接的概率，因为在这个测试里所有的音频流都是完全相同的文件
    speech_probs_conn0 = []
    
    start_time = time.time()
    
    # 时间轴模拟：随着时间推移，每一帧音频（32ms）到达服务器
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample: current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(chunk, (0, int(window_size_samples - len(chunk))))
        
        # 这个时刻，收到了 concurrency 这么多用户的当前帧
        chunks = chunk.unsqueeze(0).expand(concurrency, -1)
        
        # 将 concurrency 根据 max_batch_size 进行切片处理（这就是实际服务器上动态批大小的分发过程）
        # 如果 concurrency < max_batch_size，那么实际的 batch_size 就是 concurrency
        out_probs = []
        
        for k in range(0, concurrency, max_batch_size):
            b = min(max_batch_size, concurrency - k)
            # 提取本批次的特征、上下文和历史状态
            batch_x = chunks[k : k+b]
            batch_ctx = global_context[k : k+b]
            batch_st = global_state[:, k : k+b, :]
            
            # 拼接上下文
            batch_x = torch.cat([batch_ctx, batch_x], dim=1)
            
            # 物理并行计算
            out, new_st = raw_model(batch_x, batch_st)
            
            # 更新保存
            out_probs.append(out)
            global_context[k : k+b] = batch_x[:, -64:]
            global_state[:, k : k+b, :] = new_st
            
        # out_probs 合并，仅取第一个用户的概率作为一致性验证
        probs_all = torch.cat(out_probs, dim=0)
        speech_probs_conn0.append(probs_all[0].item())
        
    process_time = time.time() - start_time
    segments = extract_segments(speech_probs_conn0, sampling_rate=sampling_rate)
    return segments, process_time

def main():
    print("Loading audio...")
    audio_data, sr = sf.read(wav_path, dtype='float32')
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]
    
    # 截断为 10 秒以大幅加快压测速度，RTF 结论不变
    audio_tensor = torch.from_numpy(audio_data[:160000])

    print("Loading Silero VAD model...")
    silero_vad_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "vad", "silero-vad")
    model, utils = torch.hub.load(repo_or_dir=silero_vad_dir, model='silero_vad', source='local')
    torch.set_grad_enabled(False) # Ensure no grad
    
    # 建立一个基准结果（BatchSize=1, Concurrency=1）用于验证分段时间戳的一致性
    baseline_segments, _ = simulate_dynamic_batching(audio_tensor, model, 1, 1)
    
    max_batch_sizes = [64, 128, 256, 512]
    concurrencies = [100, 200, 500, 800, 1000, 1500]
    
    audio_dur_sec = len(audio_tensor) / 16000
    
    with open("dynamic_batch_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Silero VAD 单实例动态批处理 (Dynamic Batching) 压力测试\n")
        f.write(f"测试截断音频时长: {audio_dur_sec:.2f} 秒\n")
        f.write("当处理总耗时 > 测试音频时长时，判定为该配置遭遇瓶颈（即产生延迟积压）。\n\n")
        f.flush()
        
        for mb in max_batch_sizes:
            f.write(f"==========================================================\n")
            f.write(f"配置：最大批大小 Max_Batch_Size = {mb}\n")
            f.write(f"==========================================================\n")
            f.flush()
            print(f"\nTesting Max Batch Size {mb}...")
            
            for c in concurrencies:
                print(f"  Concurrency: {c}")
                segments, proc_time = simulate_dynamic_batching(audio_tensor, model, c, mb)
                
                # Check correctness
                is_correct = (len(segments) == len(baseline_segments))
                if is_correct:
                    for s1, s2 in zip(segments, baseline_segments):
                        if s1['start'] != s2['start'] or s1['end'] != s2['end']:
                            is_correct = False
                            break
                            
                status = "✅ 实时 (无积压)" if proc_time < audio_dur_sec else "❌ 瓶颈 (发生延迟积压)"
                rtf = proc_time / audio_dur_sec
                
                out_str = f"并发路数: {c:<5} | 耗时: {proc_time:>7.3f}s | RTF(系统总计): {rtf:>5.3f} | 状态: {status} | 准确率(一致性): {'100%' if is_correct else '失败'}\n"
                f.write(out_str)
                f.flush()
                print(f"    -> {out_str.strip()}")
            
            f.write("\n")
            f.flush()

if __name__ == "__main__":
    main()
