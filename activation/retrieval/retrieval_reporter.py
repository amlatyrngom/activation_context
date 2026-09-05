"""
Helps report training progress in a continually updated html file.
So every 10% of the training data, compute the reporting loss and plot it.
Should be a first class citizen, or I can't tell what's actually going on.

Required for Slice 1. @AI: The planning should actually include mocks for me to see.

The interface is likely:
__init__(path, title, description) # title and description turn into header and paragraph.
initialize_(table|line_plot|bar_plot)(name, title, description, some metadata (rows, axes))
add_data_point(name, data) # use metadata to map data to rows/axes.

@AI: Feel free to improve if this insufficiently general.

The file is updated live: In a 12h training, I should be able to know what's going on while it's happening.

Goals:
- Simple, General, Nice-to-view.
"""