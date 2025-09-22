from .calculate import calculate

# build a list of tools ready to be given over to a model.generate call
retail_tools = [
    calculate
]
