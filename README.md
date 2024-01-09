
# Jack's Car Rental Problem

## 项目简介
这个项目是关于解决Jack租车店的车辆调配问题。Jack拥有两个租车点，需要通过优化车辆调配来最大化盈利。项目采用策略迭代方法，基于马尔可夫决策过程（MDP）来求解最优调配策略。

## 问题设置
- Jack有两个租车点，每个点最多停放20辆车。
- 租出一辆车得到10美金的收入。
- 每天租出和归还的车辆数量遵循泊松分布。
- 每晚可在两个租车点间调配最多5辆车，每调配一辆车花费2美金。
- 1号租车点的租出和归还的车辆遵循λ=3的泊松分布。
- 2号租车点的租出和归还的车辆遵循λ=4和λ=2的泊松分布。
- 阻尼系数γ=0.09。

## 目录结构
```
.
├── README.md
├── jackscarrental
│   ├── __init__.py
│   ├── poisson.py
│   └── policy_iteration.py
├── requirements.txt
└── tests
    ├── test_policy_iteration.py
    └── test_policy_iteration_auto_termination.py
```

## 主要文件功能
- `poisson.py`：实现泊松分布的相关功能。
- `policy_iteration.py`：实现策略迭代算法，用于求解最优调配策略。
- `test_policy_iteration.py`：测试策略迭代算法的正确性和效率。
- `test_policy_iteration_auto_termination.py`：测试策略迭代算法的自动终止功能。

## 运行方式
首先，确保安装所有必要的依赖：
```
pip install -r requirements.txt
```
然后，可以通过以下方式运行主程序或测试：
```
python -m jackscarrental.policy_iteration
python -m pytest tests/
```

## 贡献
这是一个开放项目，欢迎任何形式的贡献和建议。

