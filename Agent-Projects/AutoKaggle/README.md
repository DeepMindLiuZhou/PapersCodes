# AutoKaggle

This is the formal repo for paper: "AutoKaggle: A Multi-Agent Framework for Autonomous Data Science Competitions"

<p align="center">
    <a href="https://m-a-p.ai/AutoKaggle.github.io/"><img src="https://img.shields.io/badge/🏠-Home Page-8A2BE2"></a>
    <a href="https://arxiv.org/abs/2410.20424.pdf"><img src="https://img.shields.io/badge/Paper-Arxiv-red"></a>
    <a href="https://github.com/multimodal-art-projection/AutoKaggle/blob/main/LICENSE.md"><img src="https://img.shields.io/badge/LICENSE-Apache--2.0-green"></a>
</p>

![kaggle_main](./mdPICs/kaggle_main.png)

## Introduction

AutoKaggle is a powerful framework that assists data scientists in completing data science pipelines through a collaborative multi-agent system. The framework combines iterative development, comprehensive testing, and a machine learning tools library to automate Kaggle competitions while maintaining high customizability. The key features of AutoKaggle include:

- **Multi-agent Collaboration**: Five specialized agents (`Reader`, `Planner`, `Developer`, `Reviewer`, and `Summarizer`) work together through six key competition phases.
- **Iterative Development and Unit Testing**: Robust code verification through debugging and comprehensive unit testing.
- **ML Tools Library**: Validated functions for data cleaning, feature engineering, and modeling.
- **Comprehensive Reporting**: Detailed documentation of workflow and decision-making processes.

![unit_test](./mdPICs/unit_test.png)

## Quick Start with AutoKaggle

### Set Environment

1. Clone the repository
```bash
git clone https://github.com/multimodal-art-projection/AutoKaggle.git
```

2. Create and activate conda environment
```bash
conda create -n AutoKaggle python=3.11
conda activate AutoKaggle
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure OpenAI API
Create `api_key.txt` with:
```
sk-xxx                           # Your API key
https://api.openai.com/v1       # Base URL
```

### Data Preparation

We support evaluation of Tabular-type datasets from Kaggle. Please Place competition data in `./multi_agents/competition/` with the following structure:
```
competition/
├── train.csv
├── test.csv
├── sample_submission.csv
└── overview.txt                 # Competition overview and data description
```

overview.txt: Copy and paste the Overview and Data sections from the Kaggle competition homepage into this file. The `Reader` will read this file to summarize relevant information.

### Running AutoKaggle

To run AutoKaggle experiments, use the following command:

```bash
bash run_multi_agent.sh
```

#### Configuration Parameters

- **Competition Selection**
  - `competitions`: Define target competitions in the script

- **Experiment Control**
  - `start_run`, `end_run`: Define experiment iterations (default: 1-5)
  - `dest_dir_param`: Output directory specification (default: "all_tools")

- **Model Configuration**
  - Default: `gpt-4o` for `Planner` and `Developer`, `gpt-4o-mini` for other agents
  - `model` determines the base model of `Planner`
  - Modify `_create_agent` in `multi_agents/sop.py` to change the base model of other agents

#### Output Structure
```
multi_agents/experiments_history/
└── <competition>/
    └── <model>/
        └── <dest_dir_param>/
            └── <run_number>/
```

## Result

We evaluated AutoKaggle across 8 diverse Kaggle competitions, achieving:
- 85% validation submission rate
- 0.82 comprehensive score

<div style="text-align: center;">
    <img src="./mdPICs/main_results.png" class="result"
    width="80%" />
    <img src="./mdPICs/average_nps.png" class="result"
    width="80%" />
</div>

## Citation

```
@misc{li2024autokagglemultiagentframeworkautonomous,
      title={AutoKaggle: A Multi-Agent Framework for Autonomous Data Science Competitions}, 
      author={Ziming Li and Qianbo Zang and David Ma and Jiawei Guo and Tianyu Zheng and Minghao liu and Xinyao Niu and Yue Wang and Jian Yang and Jiaheng Liu and Wanjun Zhong and Wangchunshu Zhou and Wenhao Huang and Ge Zhang},
      year={2024},
      eprint={2410.20424},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2410.20424}, 
}
```

## Disclaimer

This project, "AutoKaggle," is not affiliated with, endorsed by, or officially associated with Kaggle or Google in any way. The use of the name "Kaggle" is solely to indicate compatibility with Kaggle competitions. All trademarks, logos, and brand names are the property of their respective owners. We respect Kaggle's brand guidelines and are in the process of rebranding to better reflect our independence. For further details or concerns, please contact us.


## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
