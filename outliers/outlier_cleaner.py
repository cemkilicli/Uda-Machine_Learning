#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    error = []

    ### your code goes here

    ### identify and remove the most outlier-y points

    # to calculate error as specified in the code comment
    errors = (net_worths - predictions)**2
    cleaned_data = zip(ages, net_worths, errors)

    # the [0] isn't necessary in this case since errors is a 1-D array
    cleaned_data = sorted(cleaned_data, key=lambda x: x[2], reverse=True)
    limit = int(len(net_worths) * 0.1)

    # cast the iterator object as a list. I needed to do this to avoid errors
    # in the calling code.
    return list(cleaned_data[limit:])

