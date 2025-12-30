In this repository, there is code available to implement our proposed framework for improving NILM robustness based on the Diffusion model.

Our work uses Diffusion model to synthesize high-usage robust electrical load data, mix it with the original data to improve the robustness of the model, and use synthetic data to fill in the gaps when the data set is insufficient.

## Requirements:

The code requires conda3 and one CUDA capable GPU，The version of python should preferably be greater than 3.7 our environment(for reference only): tensorflow==2.3.0 pytorch==2.1.1 keras==2.4.0 scikit-learn==1.1.2 

## Dataset Preparation for UK_DALE

We use the UK_Dale low-frequency dataset(1/6Hz).UK_DALE datasets are available in (https://data.ukedc.rl.ac.uk/cgi-bin/data_browser/browse/edc/efficiency/residential/EnergyConsumption/Domestic/).

## Start electrical load data synthesis

After selecting the effective part of the electrical appliance data to be trained, you can start using the Diffusion model for load synthesis and execute the main.py file to perform load data synthesis.

Diffusion model training:

```
python main.py --name 'applianceName' --config 'applianceName.yaml' --gpu 0 --train
```

Diffusion model load synthesis:

```
python main.py --name 'applianceName' --config 'applianceName.yaml' --gpu 0 --sample 0 --milestone 'checkpoint'
```

We used LoadSynthesisExample.ipynb to perform a synthesis example.

After obtaining the synthesized load data, you can run InverseTransform.py to perform inverse normalization to obtain the complete power load data.

### Quick guide for `InverseTransform.py`
- The script needs two files:
  1. A **real** appliance power CSV (used to fit Min-Max), e.g. `home2_microwave.csv` containing a `power` column.
  2. The **synthetic** file you want to restore (``.npy`` from diffusion output or a CSV with a `power` column), e.g. `ddpm_fake_microwave.npy`.
- If your file names differ, either **rename them** to match the examples or pass the exact paths with the flags below.

Example command:
```
python InverseTransform.py \
  --appliance microwave \
  --real ./Data/home2_microwave.csv \
  --synthetic ./generatedData/ddpm_fake_microwave.npy \
  --output ./generatedData/microwave_denormalized.csv
```
If any path is wrong, the script will raise a clear error and remind you to either rename the file or pass the correct path.

If you only run `python -m py_compile InverseTransform.py`, it merely checks that the script has no syntax errors (it exits silently and writes a `__pycache__` file). To actually get a denormalized CSV and comparison plot, run the command above with `--real/--synthetic/--output` pointing to your files.

### 中文快速说明：运行 `InverseTransform.py` 之后做什么
1. **准备真实功率数据**：例如 `home2_microwave.csv`，需要有 `power` 列（用来拟合 Min-Max 缩放器）。
2. **执行反归一化**：
   ```bash
   python InverseTransform.py --appliance microwave --real 你的真实CSV路径 --synthetic 你的生成文件路径 --output generatedData/microwave_denormalized.csv
   ```
   - `--synthetic` 可以是 `.npy`（如 `ddpm_fake_microwave.npy`）或包含 `power` 列的 CSV。
   - 如果默认文件名不存在，脚本会提示你用参数传入实际路径。
3. **产出是什么**：命令结束后，会把还原到瓦特值的功率序列写入 `--output` 指定的 CSV，并弹出生成曲线 vs. 真实曲线的对比图（默认展示前 2 万个点）。
4. **之后要做什么**：如果要把合成数据用于 NILM 训练/评估，可以直接使用生成的 CSV（已恢复到真实功率尺度），按你的预处理/训练流程继续即可。

## Start the NILM testing

Put the original UK_DALE data into the folder directory dataset_preprocess and name it UK_DALE. Run uk_dale_processing.py to perform Z-Score standardization to obtain the prepared NILM original training and test datasets. We use house 2 for training and testing, and house 1 for cross-house testing.

When using the framework for NILM data augmentation, the synthetic data must first be normalized with the same Z-Score as the original data before being mixed for NILM model training.

After preprocessing all datasets, you can run NILM-main/EasyS2S_test.py to verify the results in our paper. We provide the baseline models of 5 appliances and the enhanced models ('*.h5' files), which are stored in the file directory 'NILM-main/models'



## Visualization

<img src="Figure.png" alt="image-20240714214100579" style="zoom: 50%;" />

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base:

https://github.com/linfengYang/AugLPN_NILM   
https://github.com/Y-debug-sys/Diffusion-TS  
https://github.com/LeapLabTHU/Agent-Attention  
https://github.com/MingjunZhong/NeuralNetNilm  
https://github.com/MingjunZhong/transferNILM/  

C. Zhang, M. Zhong, Z. Wang, N. Goddard, and C. Sutton. Sequence-to-point learning with neural networks for non-intrusive load monitoring. In Proceedings for Thirty-Second AAAI Conference on Artificial Intelligence. AAAI Press, 2018.

------

Contact e-mail:miles_gzy@163.com
