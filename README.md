# 🏠 Word2Walls

**Word2Walls** is a text-driven framework for generating diverse and realistic 3D room layouts directly from natural language prompts. This project builds upon the [AnyHome](https://github.com/FreddieRao/anyhome_github) paper and its official implementation, extending it with modular layout generation and improved constraint-based object placement.

---

## 📚 Based On

This project is based on the methodology introduced in the **AnyHome** paper:

> Fu, R., Wen, Z., Liu, Z., & Sridhar, S. (2024). *AnyHome: Open-Vocabulary Generation of Structured and Textured 3D Homes*. arXiv:2312.06644.  
> GitHub Repo: [https://github.com/FreddieRao/anyhome_github](https://github.com/FreddieRao/anyhome_github)

---

## 🧠 Project Overview

Designing 3D room layouts is a fundamental yet challenging task in interior design that traditionally demands extensive manual effort and expertise. Inspired by recent advances in open-vocabulary scene generation—exemplified by the AnyHome framework—this project explores a system that directly translates natural language into diverse and realistic 3D room layouts.

Our approach has two stages:  
1. **Semantic Parsing** of user text using a large language model  
2. **Generative Layout Synthesis** with spatial constraints and placement rules

The system outputs functionally plausible and visually coherent layouts based on room types, relationships, and furniture arrangements inferred from text.

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/huygnguyen04/word2walls.git
cd anyhome_github
```

## 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

Note: You may need to install system packages like Graphviz. See https://pygraphviz.github.io/documentation/stable/install.html if needed
```bash
# Ubuntu
sudo apt-get update
sudo apt-get install python3-dev
sudo apt-get install graphviz libgraphviz-dev
pip install pygraphviz
```

## 3. Set Your OpenAI API Key 
You can add it in credentials.py:
```
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

## 4. Modify the Prompt (Optional)
Edit the prompt in main.py:
```
prompt = "a 1B1B house with a garage and a kitchen"
```

## 5. Run the Pipeline
```
python main.py
```
This will generate your layout and optionally visualize or refine it.






