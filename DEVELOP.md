# Developing

To update the documentation: 

    pydoc-markdown -I divexplorer -m divexplorer -m outcomes -m pattern_processor -m shapley_value > Documentation.md
    cat README_prototype.md Documentation.md > README.md

To upload to pypi: 

    python3 -m build
    python3 -m twine upload dist/*

To install locally: 

    python3 -m pip install . 

