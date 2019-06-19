From command prompt (Run as administrator)

D:\projects\datascience\kesh-utils>python setup.py sdist bdist_wheel

Once package successfully build - check the dist folder


# Upload to https://pypi.org/project/

D:\projects\datascience\kesh-utils>python -m twine upload dist/*

(1xS....!$)