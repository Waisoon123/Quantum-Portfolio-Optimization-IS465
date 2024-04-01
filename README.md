# Quantum-Portfolio-Optimization-IS465
Quantum Portfolio Optimization Demo Application

## Table of Contents:
- [Introduction](#intro)
- [Project Setup & Requirements](#setup)
- [References](#reference)

<h3 id="intro">Introduction</h3>

This project leverages quantum computing to optimize portfolio management in the financial sector. Utilizing D-Wave's quantum annealing technology, we aim to outperform traditional portfolio optimization methods such as Modern Portfolio Theory (MPT) by maximizing returns and minimizing risks. The solution integrates advanced quantum algorithms with financial data from Yahoo Finance, focusing on S&P 500 stocks. Our approach aims to demonstrates the potential of quantum computing in enhancing financial decision-making and offers insights into the practical application of quantum technologies in portfolio optimization.

<h3 id="setup">Project Setup & Requirements</h3>

Pre-requisites : Note that you may need to create an account on DWAVE - https://cloud.dwavesys.com/leap/signup/ to access the API token needed, in case the API token used in the source code has expired as this account is under a free trial with limited resources. You can find the token on the dashboard view of the website, after successfully signing up for an free trial account.

1. Clone Repository from Github Desktop
2. Open cmd in directory of the cloned files / open in visual studio code terminal
3. Create a virtual environment in the root directory with the command "Python -m venv venv"
4. Pip install all required modules before running the source code with the command "pip install -r requirements.txt'
5. Run the command "py quantumDemo.py" to run the source code
6. Alternatively, you may run the jupyter notebook from the directory using the command "jupyter notebook".



<h3 id="reference">References</h3>

https://towardsdatascience.com/exploration-of-quantum-computing-solving-optimisation-problem-using-quantum-annealer-77c349671969
https://github.com/dwave-examples/portfolio-optimization
https://medium.com/mdr-inc/portfolio-optimization-with-minimum-risk-using-qaoa-e29e1d66c194
https://colab.research.google.com/gist/kenichi-lon/fd086ab3e9bea539d92470e48493934b/tds_qa_example_portfolio_optimisation.ipynb
