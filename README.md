# Compare the impact of fairness analysis on model training
Apply the SURE fairness analysis algorithm to graph data, train a GNN model to handle the graph dataset, and compare the impact on performance between models with and without fairness analysis.

### Run the code
- 執行包含公平性分析的 GNN 模型：
```bash
python newtest_GNN.py
```
- 執行不包含公平性分析的 GNN 模型：
```bash
python newtest_noSure.py
```

### Reference:
<pre>
SURE: Robust, Explainable, and Fair Classification without Sensitive Attributes,
by D. Chakrabarti,
in KDD 2023.
</pre>
