# IPL Cricket Prediction System

An advanced machine learning system for predicting IPL match outcomes, player performances, and generating natural language explanations using LLM.

## Features

- **Match Winner Prediction**: Ensemble model combining XGBoost, Random Forest, and Neural Network
- **Score Prediction**: Predicts final scores for both teams
- **Player Performance Analysis**: Tracks key metrics like runs, wickets, and economy rate
- **Performance Trends**: Analyzes team and player performance over recent matches
- **LLM Integration**: Provides natural language explanations for predictions
- **Weather Impact Analysis**: Considers weather conditions and pitch state
- **Player Availability Tracking**: Monitors key player availability

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sumithkumar07/IPL-cricket-prediction-system.git
cd IPL-cricket-prediction-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama and pull the llama2 model:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama2
```

## Project Structure

```
IPL-cricket-prediction-system/
├── ml_model/
│   ├── ensemble.py          # Ensemble model implementation
│   ├── llm_reasoning.py     # LLM integration for explanations
│   └── time_series.py       # Time series analysis
├── data/
│   ├── raw/                 # Raw data
│   └── processed/           # Processed data
├── models/                  # Trained models
├── tests/                   # Test scripts
├── requirements.txt         # Dependencies
└── README.md               # Documentation
```

## Usage

1. Train the models:
```bash
python train_models.py
```

2. Test the prediction system:
```bash
python test_llm_integration.py
```

3. Generate predictions with explanations:
```python
from ml_model.llm_reasoning import IPLPredictionExplainer

explainer = IPLPredictionExplainer()
result = explainer.predict_winner(match_data)
print(result['explanation'])
```

## Model Performance

- Match Winner Prediction Accuracy: 96.69%
- Score Prediction MAE: 0.42 runs
- Player Performance Prediction R²: Varies by metric

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- IPL data sources
- Machine learning libraries
- Ollama for LLM integration 