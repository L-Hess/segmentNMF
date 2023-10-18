import setuptools

setuptools.setup(
    name="segmentNMF",
    version="0.0.1",
    author="Luuk Hesselink",
    author_email="",
    description="",
    url="",
    license="",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.20.3',
        'scipy>=1.9.1',
        'ClusterWrap>=0.3.0',
        'zarr>=2.12.0',
        'numcodecs>=0.9.1',
        'pynrrd',
        'dask',
        'dask[distributed]',
        'tqdm',
        'torch',
    ]
)
