from setuptools import setup, find_packages

setup(
  name = 'agent-attention-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.1.6',
  license='MIT',
  description = 'Agent Attention - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/agent-attention-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'attention',
    'linear attention'
  ],
  install_requires=[
    'einops>=0.7.0',
    'torch>=2.0'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
