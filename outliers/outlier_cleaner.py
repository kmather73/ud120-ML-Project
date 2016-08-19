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

    ### your code goes here
    for i in range(len(ages)):
        cleaned_data.append( (ages[i][0], \
                              net_worths[i][0], \
                              (predictions[i][0]-net_worths[i][0])**2))
    
    cleaned_data.sort(cmp=lambda a,b: -1 if (a[2]<b[2]) else 1)
    return cleaned_data[0: int(len(ages)*.9)]

