from setuptools import setup, find_packages

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He, Ziqin Rong'
__email__ = 'tanjin_he@berkeley.edu, rongzq08@gmail.com'

if __name__ == "__main__":
    setup(name='materials_entity_recognition',
          version=3.3,
          author="Tanjin He",
          author_email="tanjin_he@berkeley.edu",
          license="MIT License",
          packages=find_packages(),
          include_package_data=True,
          install_requires=[
              'tensorflow==2.7.0',
              'tensorflow-addons==0.17.1',
              'protobuf==3.19.6',
              'spacy',
              'chemdataextractor',
              'numpy',
              'transformers>=4.11.3',
              'torch',
              'psutil',
          ],
          zip_safe=False)
