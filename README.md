# 🏠 Word2Walls

## 📑 [Technical Report](https://docs.google.com/presentation/d/1nJhuzPO8a2mRjVUCf0XmAqj9vl_Ds8dUVjueYs-B86g/edit?usp=sharing) & 📊 [Presentation Slides](https://drive.google.com/file/d/1h9IplaxKWE1u3uldueNFlSDVMDEdJWIq/view?usp=sharing)

<br>

**Word2Walls** is a text-driven framework for generating diverse and realistic 3D room layouts directly from natural language prompts. This project builds upon the [AnyHome](https://github.com/FreddieRao/anyhome_github) framework, extending it with modular layout generation and improved constraint-based object placement.

## ✨ Key Features

- 🏢 **Floorplan Generation** - Create house floorplans from natural language descriptions
- 🛋️ **Furniture Placement** - Automatic layout optimization with spatial constraints
- 🎨 **Texture Generation** - Generate realistic room textures for visualization
- 📷 **3D Rendering** - Photorealistic visualization of generated spaces

## 📚 Based On

This project extends the methodology introduced in the **AnyHome** paper:

> Fu, R., Wen, Z., Liu, Z., & Sridhar, S. (2024). *AnyHome: Open-Vocabulary Generation of Structured and Textured 3D Homes*. arXiv:2312.06644.  
> GitHub Repo: [https://github.com/FreddieRao/anyhome_github](https://github.com/FreddieRao/anyhome_github)

## 🧠 Project Overview

Word2Walls transforms natural language descriptions into fully realized 3D environments through a multi-stage pipeline:

1. **Semantic Parsing** - Extracts room types, relationships, and furniture requirements from text
2. **Floorplan Generation** - Creates a structured house layout with proper room relationships
3. **Furniture Placement** - Places furniture objects within rooms based on functional requirements
4. **Layout Optimization** - Refines object placement using spatial constraint satisfaction
5. **Texture Generation** - Applies realistic textures to walls, floors, and objects
6. **3D Rendering** - Produces photorealistic visualizations of the generated space

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/word2walls.git
cd word2walls
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. System Dependencies

For the full experience, additional system packages may be required:

#### Graphviz (for visualization)
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev graphviz libgraphviz-dev

# Windows (using Chocolatey)
choco install graphviz

# macOS
brew install graphviz
```

#### OpenGL Support (for 3D rendering)
```bash
# Ubuntu/Debian
sudo apt-get install freeglut3-dev

# Windows
# OpenGL libraries are typically included with graphics drivers

# macOS
# OpenGL is pre-installed
```

## 🔑 API Key Setup

Set your OpenAI API key in `credentials.py`:
```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

## 🚀 Usage

### Basic Usage

Run with default settings (generates a bedroom):
```bash
python main.py
```

### Custom Room Generation

Specify your own room description:
```bash
python main.py --prompt "A spacious living room with a sectional sofa, coffee table, and large windows overlooking a garden"
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--prompt "TEXT"` | Text description of the room to generate |
| `--edit` | Enable interactive editing of the floorplan and layout |
| `--no-3d` | Disable 3D rendering (faster generation) |
| `--skip-optimization` | Skip the layout optimization phase |

### Example Commands

Generate with interactive editing:
```bash
python main.py --prompt "A modern kitchen with an island" --edit
```

Generate without 3D rendering (faster):
```bash
python main.py --prompt "A home office with a desk" --no-3d
```

## 📁 Output Files

Generated files are saved in the `output/` directory:
- `output/floorplan/`: Floor plan visualizations
- `output/layout/`: Room layout visualizations
- `output/optimization/`: Layout optimization results
- `output/texture/`: Generated textures
- `output/3d_renders/`: 3D rendered images
- `output/3d_textures/`: Textured 3D models

## 📝 License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

Copyright (c) 2024 Huy Nguyen and Santiago Perez

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- This project builds upon the work of the AnyHome paper authors
- Special thanks to all contributors and testers






