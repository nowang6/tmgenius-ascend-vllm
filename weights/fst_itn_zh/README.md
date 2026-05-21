---
license: Apache License 2.0
tasks:
- inverse-text-processing
---
基于openfst构建的，自定义规则的ITN模型。
使用[WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)构建。
# 模型下载
```bash
 git clone https://www.modelscope.cn/thuduj12/fst_itn_zh.git
```

# 模型生成
如果想使用原始开源版本定义的规则，可以使用如下代码
```bash 
git clone https://github.com/wenet-e2e/WeTextProcessing.git
cd WeTextProcessing
python -m itn --text "二点五平方电线" --overwrite_cache
```
运行后在WeTextProcessing/itn下会生成两个fst文件。

如果需要自定义规则，则需要修改WeTextProcessing/itn/chinese/xxx.py文件，
然后再基于新的规则，重新构建，并覆盖之前构建的模型。

# 模型使用

## Python中使用
```python
>>> from itn.chinese.inverse_normalizer import InverseNormalizer
>>> invnormalizer = InverseNormalizer(cache_dir="/path/to/your/itn")
>>> result = invnormalizer.normalize("二点五平方电线")
>>> print(result)
```

## Cpp Runtime
目前对[FunASR](https://github.com/alibaba-damo-academy/FunASR)和[Wenet](https://github.com/wenet-e2e/wenet)的支持还在开发中，
### FunASR
可以参考 https://github.com/alibaba-damo-academy/FunASR/pull/884

### Wenet
可以参考 https://github.com/wenet-e2e/wenet/pull/2001