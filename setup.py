import setuptools

setuptools.setup(
    name="base_script",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas"
    ],
    entry_points={'console_scripts':
                  ['script = main:main']}
)
