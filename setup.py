import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "augmento",
    version = "0.0.1",
    author = "Syed Muhammad Asad",
    author_email = "asdkazmi@gmail.com",
    description = "Augmento is a Python library that provides a collection of image augmentation techniques for machine learning applications.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/rev9ai/augmento",
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">=3.6",
    setup_requires=["numpy"],
    install_requires=["numpy", "opencv-python"]
)