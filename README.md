# EasyHPC

[![Documentation Status](https://readthedocs.org/projects/easyhpc/badge/?version=latest)](https://easyhpc.readthedocs.io/en/latest/?badge=latest)
[![Docs build](https://github.com/pescap/EasyHPC/actions/workflows/docs.yml/badge.svg)](https://github.com/pescap/EasyHPC/actions/workflows/docs.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/2cb491b027e44d519e34ffd5116ec902)](https://www.codacy.com/gh/pescap/EasyHPC/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=pescap/EasyHPC&amp;utm_campaign=Badge_Grade)
[![License](https://img.shields.io/github/license/pescap/EasyHPC)](https://github.com/pescap/EasyHPC/blob/main/LICENSE)

A practical guide to your first High Performance Computing (HPC) project, with a focus on deep learning and physics-informed neural networks.

Read it at **[easyhpc.readthedocs.io](https://easyhpc.readthedocs.io)**.

## What you will learn

- GitHub: forks, branches, issues and pull requests
- CI/CD: reStructuredText, Sphinx, GitHub Actions and Read the Docs
- System administration: the Linux shell, SSH, Jupyter on a remote server, GPUs, conda and Docker
- Distributed deep learning with Horovod
- Physics-informed neural networks with DeepXDE
- Neural network theory

## Contributing

Issues and pull requests are welcome. See the [contributing guide](https://easyhpc.readthedocs.io/en/latest/contributing.html). To build the documentation locally:

```bash
pip install -r requirements.txt
sphinx-build -b html docs/source docs/build/html
```

## Credits

EasyHPC started in 2022 as the shared documentation of an undergraduate research course at Universidad Adolfo Ibáñez, Santiago, Chile. See the [credits](https://easyhpc.readthedocs.io/en/latest/credits.html) for the people who wrote it.
