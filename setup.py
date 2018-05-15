from distutils.core import setup

setup(name='sume',
      version='1.0',
      description='sume is a summarization library written in Python.',
      author='Florian Boudin',
      author_email='florian.boudin@univ-nantes.fr',
      license='gnu',
      packages=['sume', 'sume.models'],
      url="https://github.com/boudinfl/sume",
      install_requires=[
      	'nltk',
      	'numpy'
      ]
     )