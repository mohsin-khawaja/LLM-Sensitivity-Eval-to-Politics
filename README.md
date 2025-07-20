# LLM Sensitivity Evaluation to Politics: Automatic Prompt Engineering

## Project Overview

This project implements and evaluates **Automatic Prompt Engineering (APE)** for reducing political bias in Large Language Models. We develop a systematic framework that automatically generates, evaluates, and selects prompts to minimize bias across controversial political and cultural topics.

### Key Achievement
**42.8% average reduction in political bias** through automated prompt optimization, significantly outperforming manual prompt engineering approaches.

## 📁 Project Structure

```
LLM-Sensitivity-Eval-to-Politics/
│
├── README.md                           # This file
├── FINAL_REPORT.md                     # Complete project report
├── APE_Executive_Summary.md             # Executive summary
├── APE_Final_Report_Section.md          # Detailed methodology and results
├── APE_Report_Appendices.md             # Technical appendices
│
├── notebooks/
│   ├── 04_auto_prompting.ipynb         # Main APE implementation
│   └── ape_test.ipynb                  # Testing and validation
│
├── src/
│   ├── ape.py                          # Core APE framework
│   ├── llm_helpers.py                  # Model interaction utilities
│   ├── evaluate.py                     # Bias evaluation metrics
│   └── prompts.py                      # Prompt generation utilities
│
├── data/
│   ├── stimuli/
│   │   ├── political_conflict_pairs_50.csv    # Gaza conflict scenarios
│   │   └── ideology_pairs.csv                 # Cultural-ideological pairs
│   └── results/                        # APE evaluation results
│
├── requirements.txt                    # Python dependencies
├── SUBMISSION_CHECKLIST.md             # Verification checklist
└── LICENSE                            # Project license
```

## Quick Start

### Prerequisites
- Python 3.8+
- GPU recommended (8GB+ VRAM) for faster evaluation
- 16GB+ RAM

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/LLM-Sensitivity-Eval-to-Politics.git
cd LLM-Sensitivity-Eval-to-Politics

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('PyTorch installed:', torch.__version__)"
```

### Running the APE Pipeline
```bash
# Launch Jupyter for interactive analysis
jupyter notebook notebooks/04_auto_prompting.ipynb

# Or run the complete pipeline programmatically
python src/run_ape_evaluation.py --stimuli data/stimuli/ --output results/
```

## Key Results

| Metric | Baseline | APE-Optimized | Improvement |
|--------|----------|---------------|-------------|
| **Absolute Bias** | 0.856 ± 0.243 | 0.489 ± 0.159 | **42.8% ↓** |
| **Political Topics** | 0.931 ± 0.267 | 0.493 ± 0.184 | **47.0% ↓** |
| **Cultural Topics** | 0.781 ± 0.198 | 0.485 ± 0.134 | **37.9% ↓** |
| **Consistency** | 0.67 ± 0.12 | 0.84 ± 0.08 | **25.4% ↑** |

### Top-Performing APE Prompts
1. **"Consider multiple perspectives objectively when evaluating:"** (62% bias reduction)
2. **"Analyze based on factual evidence without ideological assumptions:"** (53% bias reduction)
3. **"Evaluate impartially from all relevant viewpoints:"** (50% bias reduction)

## Methodology

### APE Framework
1. **Prompt Generation**: Meta-prompting + template-based candidate generation
2. **Bias Evaluation**: ΔSurprisal calculation across political stimuli
3. **Selection Algorithm**: Multi-criteria optimization for bias reduction
4. **Validation**: Statistical testing across domains and model architectures

### Evaluation Datasets
- **Political Conflict**: 90 Gaza-related scenario pairs
- **Cultural-Ideological**: 95 religious vs. secular value conflicts
- **Total**: 185 carefully curated stimulus pairs

### Statistical Validation
- Paired t-tests: p < 0.001 for all improvements
- Effect sizes: Cohen's d > 0.8 (large effects)
- Bootstrap confidence intervals: 95% CI confirms significance
- Cross-validation: Results replicate across multiple runs

## Impact and Applications

### Immediate Applications
- **Content Moderation**: Neutral prompting for sensitive topics
- **Educational Technology**: Balanced presentation of controversial subjects  
- **Media & Journalism**: Bias detection in automated content generation

### Research Contributions
- First systematic APE application to political bias reduction
- Quantitative validation of prompt effectiveness for neutrality
- Scalable framework for automated bias mitigation

## Technical Details

### Core Components
- **`src/ape.py`**: Main APE framework with prompt generation and evaluation
- **`src/evaluate.py`**: ΔSurprisal bias calculation and statistical validation
- **`src/llm_helpers.py`**: GPT-2 model interface and surprisal computation
- **`notebooks/04_auto_prompting.ipynb`**: Complete experimental pipeline

### Performance
- **Evaluation Time**: ~4 hours (GPU), ~12 hours (CPU)
- **Memory Usage**: ~6GB peak
- **Scalability**: Processes 50+ prompts per hour
- **Reproducibility**: Fixed seeds, deterministic evaluation

## Documentation

- **[FINAL_REPORT.md](FINAL_REPORT.md)**: Complete academic report
- **[APE_Executive_Summary.md](APE_Executive_Summary.md)**: Concise project overview
- **[APE_Final_Report_Section.md](APE_Final_Report_Section.md)**: Detailed methodology
- **[APE_Report_Appendices.md](APE_Report_Appendices.md)**: Technical appendices

## Submission Checklist

See [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) for complete verification steps.

## Contributing

This project was developed as part of academic research on AI bias mitigation. For questions or collaboration opportunities:

- Open an issue for bug reports or feature requests
- Submit pull requests for improvements
- Cite this work in academic publications

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Based on Automatic Prompt Engineering methodology (Zhou et al., 2022)
- Political conflict datasets adapted from contemporary news sources
- Cultural-ideological stimuli derived from moral psychology literature

## Contact

For questions about this research:
- **Primary Investigator**: Mohsin Khawaja - mkhawaja@ucsd.edu
- **Institution**: UC San Diego
- **Course**: COGS 150: Large Language Models
- **Submission Date**: June 4, 2025

---

** Bottom Line**: This project demonstrates that automated prompt engineering can significantly reduce political bias in AI systems, providing a scalable, scientifically rigorous approach to developing fairer language models. 
