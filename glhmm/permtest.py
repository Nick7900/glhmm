import numpy as np
from tqdm import tqdm
import random
from scipy.stats import pearsonr
np.seterr(divide='ignore', invalid='ignore')


def between_subject_test(X_data, y_data, idx_data=None, method="regression", Nperm=1000, confounds = None, exchangeable=True,test_statistic_option=False):
    """
    Perform between-subject permutation testing.
    This function conducts statistical tests (regression, correlation, or correlation_com) between two datasets, `X_data`
    representing the measured data and `y_data` representing the dependent-variable, across different subjects using
    permutation testing. 
    The goal is to assess the statistical significance of relationships the measured data and
    the dependent variable in a between-subject design.

    Parameters:
    --------------
        X_data (numpy.ndarray): Input data array of shape that can be either a 2D array or a 3D array.
                                For 2D array, it got a shape of (n_ST, n_features), where n_ST represent 
                                the number of subjects or trials, and each column represents a feature (e.g., brain
                                region)
                                For a 3D array,it got a shape (n_timepoints, n_ST, n_features), where the first dimension 
                                represents timepoints, the second dimension represents the number of subjects or trials, 
                                and the third dimension represents features. 
                                In the latter case, permutation testing is performed per timepoint for each subject.              
        y_data (numpy.ndarray): The dependent-variable can be either a 2D array or a 3D array. 
                                For 2D array, it got a shape of (n_ST, n_predictions), where n_ST represent 
                                the number of subjects or trials, and each column represents a dependent variable
                                For a 3D array,it got a shape (n_timepoints, n_ST, n_predictions), where the first dimension 
                                represents timepoints, the second dimension represents the number of subjects or trials, 
                                and the third dimension represents a dependent variable                    
        idx_data (ndarray):     It can take index data of shape (n_trials, 2) indicating start and end indices of trials or 
                                an 1D array of shape (n_ST,) where the indices are defined for each subject/trial .
                                    Required if exchangeable=True. Defaults to None.     
        method (str, optional): The statistical method to be used for the permutation test. Valid options are
                                "regression", "correlation", or "correlation_com". (default: "regression").                                               
        Nperm (int): Number of permutations to perform (default: 1000).
        confounds (numpy.ndarray or None, optional): 
                                The confounding variables to be regressed out from the input data (X_data).
                                If provided, the regression analysis is performed to remove the confounding effects. 
                                (default: None):                               
        exchangeable (bool, optional): If True, the function performs exchangeable permutation between subjects based
                                       on the provided `idx_data`. (default: False).
                                       
        test_statistic_option (bool, optional): 
                                If True, the function will return the test statistic for each permutation.
                                (default: False) 
                                
    Returns:
    ----------  
                                Depending on the `test_statistic_option` and `method`, it can return the p-values, 
                                correlation coefficients, test statistics.
                                pval (numpy array): p-values for the test (n_timepoints, n_features) if method=="Regression", else (n_timepoints, n_features, n_predictions).
                                corr_coef (numpy array): Correlation Coefficients for the test n_timepoints, n_features, n_predictions) if method=="correlation or "correlation_com", else None.
                                test_statistic_list (numpy array): Test statistic values (n_timepoints, Nperm, n_features) if test_statistic_option is True, else None.
                                pval_list (numpy array): P-values for each time point (n_timepoints, Nperm, n_features) if test_statistic_option is True and method is "correlation_com", else None.
        

    Note:
        - The function automatically determines whether permutation testing is performed per timepoint for each subject or
          for the whole data based on the dimensionality of `X_data`.
        - The function assumes that the number of rows in `X_data` and `y_data` are equal
                                  
        
    Example:
        X_data = np.random.rand(100, 3)  # Simulated brain activity data (3 features)
        y_data = np.random.rand(100, 1)  # Simulated dependent variable data (1 variable)
        pval, test_statistic_list = between_subject_test(X_data, y_data, method="regression", Nperm=1000,
                                                         confounds=None, exchangeable=True, test_statistic_option=True)
        print("P-values:", pval)
        print("Test Statistics:", test_statistic_list)
    """
    

    # Check validity of method and data_type
    valid_methods = ["regression", "correlation", "correlation_com"]
    check_value_error(
        method in valid_methods, "Invalid option specified for 'method'. Must be one of: " + ', '.join(valid_methods)
    )
    # idx_trial is by default None
    idx_trial = None
    if exchangeable==False:
        if idx_data is None:
            raise ValueError("Warning: Indices for each subject are not provided, prohibiting permutation between subjects when exchangeable is False.")
        # Get indices for permutation
        if len(idx_data.shape)==2:
            idx_array = get_indices_array(idx_data)
            
        n_trial_subject, trial_per_subject = np.unique(idx_data, return_counts=True)
        if len(set(trial_per_subject)) != 1:
            raise ValueError("Warning: Unequal number of trials per subject prohibs permutation between subjects when exchangeable is False.")
        # Get the number of trials per subject
        idx_trial = np.arange(0, trial_per_subject[0] * len(n_trial_subject), trial_per_subject[0])

    n_timepoints, n_ST, n_features, X_data, y_data = get_input_shape(X_data, y_data)
    n_predictions = y_data.shape[-1]


    # Initialize arrays based on shape of data shape and defined options
    pval, corr_coef, test_statistic_list, pval_list = initialize_arrays(X_data, n_features, n_predictions, n_timepoints, method, Nperm, test_statistic_option)

    for t in tqdm(range(n_timepoints)) if n_timepoints > 1 else range(n_timepoints):
        # Create test_statistic and pval_perms based on method
        test_statistic, pval_perms, proj = initialize_permutation_matrices(method, Nperm, n_features, n_predictions, y_data[t, :])

        # If confounds exist, perform confound regression
        X_t = calculate_X_t(X_data[t, :], confounds)

        # Get indices for permutation
        permute_idx_list = between_subject_indices(Nperm, X_t, idx_trial, exchangeable)

        #for perm in range(Nperm):
        for perm in tqdm(range(Nperm)) if n_timepoints == 1 else range(n_timepoints):
            # Perform permutation on X_t
            Xin = X_t[permute_idx_list[perm]]
            test_statistic, pval_perms = test_statistic_calculations(Xin, y_data[t, :], perm, pval_perms, test_statistic, proj, method)

        pval, corr_coef = get_pval(test_statistic, pval_perms, Nperm, method, t, pval, corr_coef)
        # Output test statistic if it is set to True can be hard for memory otherwise
        if test_statistic_option==True:
            test_statistic_list[t,:] = test_statistic
            #  if pval_perms is empty (evaluates to False), the right-hand side of the assignment will be pval_list[t, :] itself, meaning that the array will remain unchanged.
            pval_list[t, :] = pval_perms if np.any(pval_perms) else pval_list[t, :]
    pval =np.squeeze(pval)
    corr_coef =np.squeeze(corr_coef)   
    test_statistic_list =np.squeeze(test_statistic_list)   
    pval_list =np.squeeze(pval_list)      
    # Return values if test_statistic_option== True
    if test_statistic_option == True and method =="regression":   
        return pval, test_statistic_list
    elif test_statistic_option == True and method =="correlation":   
        return corr_coef, test_statistic_list
    elif test_statistic_option == True and method =="correlation_com":   
        return pval, corr_coef, test_statistic_list, pval_list
    elif method =="regression":
        return pval
    elif method =="correlation":
        return corr_coef
    elif method =="correlation_com":
        return pval, corr_coef


def within_session_between_trial_test(X_data, y_data, idx_data, method="regression", Nperm=1000, confounds=None,test_statistic_option=False):
    """
    This function conducts statistical tests (regression, correlation, or correlation_com) between two datasets, `X_data`
    representing the measured data  and `y_data` representing the dependent-variable, within a session across different
    trials using permutation testing. The goal is to assess the statistical significance of relationships between brain
    activity and the dependent variable within the same session but across different trials.


    Parameters:
    --------------
        X_data (numpy.ndarray): Input data array of shape that can be either a 2D array or a 3D array.
                                For 2D array, it got a shape of (n_ST, n_features), where n_ST represent 
                                the number of subjects or trials, and each column represents a feature (e.g., brain
                                region)
                                For a 3D array,it got a shape (n_timepoints, n_ST, n_features), where the first dimension 
                                represents timepoints, the second dimension represents the number of subjects or trials, 
                                and the third dimension represents features. 
                                In the latter case, permutation testing is performed per timepoint for each subject.              
        y_data (numpy.ndarray): The dependent-variable can be either a 2D array or a 3D array. 
                                For 2D array, it got a shape of (n_ST, n_predictions), where n_ST represent 
                                the number of subjects or trials, and each column represents a dependent variable
                                For a 3D array,it got a shape (n_timepoints, n_ST, n_predictions), where the first dimension 
                                represents timepoints, the second dimension represents the number of subjects or trials, 
                                and the third dimension represents a dependent variable                    
        idx_data (numpy.ndarray): The indices for each trial within the session. It should be a 2D array where each row
                                  represents the start and end index for a trial.    
        method (str, optional): The statistical method to be used for the permutation test. Valid options are
                                "regression", "correlation", or "correlation_com". (default: "regression").
        Nperm (int): Number of permutations to perform (default: 1000).
        confounds (numpy.ndarray or None, optional): 
                                The confounding variables to be regressed out from the input data (X_data).
                                If provided, the regression analysis is performed to remove the confounding effects. 
                                (default: None):                                                              
        test_statistic_option (bool, optional): 
                                If True, the function will return the test statistic for each permutation.
                                (default: False) 
                                
                                
                                
    Returns:
    ----------  
                                Depending on the `test_statistic_option` and `method`, it can return the p-values, 
                                correlation coefficients, test statistics.
                                pval (numpy array): p-values for the test (n_timepoints, n_features) if method=="Regression", else (n_timepoints, n_features, n_predictions).
                                corr_coef (numpy array): Correlation Coefficients for the test n_timepoints, n_features, n_predictions) if method=="correlation or "correlation_com", else None.
                                test_statistic_list (numpy array): Test statistic values (n_timepoints, Nperm, n_features) if test_statistic_option is True, else None.
                                pval_list (numpy array): P-values for each time point (n_timepoints, Nperm, n_features) if test_statistic_option is True and method is "correlation_com", else None.
        

    Note:
        - The function automatically determines whether permutation testing is performed per timepoint for each subject or
          for the whole data based on the dimensionality of `X_data`.
        - The function assumes that the number of rows in `X_data` and `y_data` are equal
                                
    Example:
        X_data = np.random.rand(100, 3)  # Simulated brain activity data (3 features)
        y_data = np.random.rand(100, 1)  # Simulated dependent variable data (1 variable)
        idx_data = np.array([[0, 49], [50, 99]])  # Two trials within the session
        pval, test_statistic_list = within_session_between_trial_test(X_data, y_data, idx_data, method="correlation",
                                                                     Nperm=1000, confounds=None,
                                                                     test_statistic_option=True)
        print("P-values:", pval)
        print("Test Statistics:", test_statistic_list)
    """
    
    # Check validity of method
    valid_methods = ["regression", "correlation", "correlation_com"]
    check_value_error(method in valid_methods, "Invalid option specified for 'method'. Must be one of: " + ', '.join(valid_methods))

    # Get input shape information
    n_timepoints, n_ST, n_features, X_data, y_data = get_input_shape(X_data, y_data)
    n_predictions = y_data.shape[-1]
    
    # Get indices for permutation
    if len(idx_data.shape)==2:
        idx_array = get_indices_array(idx_data)
            

    # Initialize arrays based on shape of data shape and defined options
    pval, corr_coef, test_statistic_list, pval_list = initialize_arrays(X_data, n_features, n_predictions, n_timepoints, method, Nperm, test_statistic_option)


    for t in tqdm(range(n_timepoints)) if n_timepoints > 1 else range(n_timepoints):
        # Create test_statistic and pval_perms based on method
        test_statistic, pval_perms, proj = initialize_permutation_matrices(method, Nperm, n_features, n_predictions, y_data[t, :])

        # If confounds exist, perform confound regression
        X_t = calculate_X_t(X_data[t, :], confounds)

        for perm in range(Nperm):
        #for perm in tqdm(range(Nperm)) if n_timepoints == 1 else range(n_timepoints):
            # Perform permutation on X_t
            Xin = within_session_between_trial(X_t, idx_array, perm)
            test_statistic, pval_perms = test_statistic_calculations(Xin, y_data[t, :], perm, pval_perms, test_statistic, proj, method)

        pval, corr_coef = get_pval(test_statistic, pval_perms, Nperm, method, t, pval, corr_coef)
        if test_statistic_option==True:
            test_statistic_list[t,:] = test_statistic
            #  if pval_perms is empty (evaluates to False), the right-hand side of the assignment will be pval_list[t, :] itself, meaning that the array will remain unchanged.
            pval_list[t, :] = pval_perms if np.any(pval_perms) else pval_list[t, :]
    pval =np.squeeze(pval)
    corr_coef =np.squeeze(corr_coef)   
    test_statistic_list =np.squeeze(test_statistic_list)   
    pval_list =np.squeeze(pval_list)           
    # Return values if test_statistic_option== True
    if test_statistic_option == True and method =="regression":   
        return pval, test_statistic_list
    elif test_statistic_option == True and method =="correlation":   
        return corr_coef, test_statistic_list
    elif test_statistic_option == True and method =="correlation_com":   
        return pval, corr_coef, test_statistic_list, pval_list
    elif method =="regression":
        return pval
    elif method =="correlation":
        return corr_coef
    elif method =="correlation_com":
        return pval, corr_coef
   

def within_session_continuous_test(vpath_data, y_data, n_states, method="regression", Nperm=1000, test_statistic_option=False):
    """
    Perform permutation testing within a session for continuous data.

    This function conducts statistical tests (regression, correlation, or correlation_com) between a hidden state path
    (`vpath_data`) and a dependent variable (`y_data`) within each session using permutation testing. The goal is to
    assess the statistical significance of relationships between the hidden state path and the dependent variable.

    Parameters:
    --------------
        vpath_data (numpy.ndarray): The hidden state path data. It could be a 2D array where each row represents a
                                    timepoint and each column represents a state variable of shape (n_timepoints, n_states) 
                                    or a 1D array of of shape (n_timepoints,) where each row value represent a giving state.        
        y_data (numpy.ndarray): The dependent-variable with a shape of (n_ST, n_predictions), where n_ST represent 
                        the number of subjects or trials, and each column represents a dependent variable
                        For a 3D array,it got a shape (n_timepoints, n_ST, n_predictions), where the first dimension 
                        represents trials for each timepoint and each column represents a dependent variable                 
        n_states (int): The number of hidden states in the hidden state path data.
        method (str, optional): The statistical method to be used for the permutation test. Valid options are
                                "regression", "correlation", or "correlation_com". (default: "regression").
        Nperm (int): Number of permutations to perform (default: 1000).
        confounds (numpy.ndarray or None, optional): 
                                The confounding variables to be regressed out from the input data (X_data).
                                If provided, the regression analysis is performed to remove the confounding effects. 
                                (default: None):     
                                
                                
    Returns:
    ----------  
                                Depending on the `test_statistic_option` and `method`, it can return the p-values, 
                                correlation coefficients, test statistics.
                                pval (numpy array): p-values for the test (n_timepoints, n_features) if method=="Regression", else (n_timepoints, n_features, n_predictions).
                                corr_coef (numpy array): Correlation Coefficients for the test n_timepoints, n_features, n_predictions) if method=="correlation or "correlation_com", else None.
                                test_statistic_list (numpy array): Test statistic values (n_timepoints, Nperm, n_features) if test_statistic_option is True, else None.
                                pval_list (numpy array): P-values for each time point (n_timepoints, Nperm, n_features) if test_statistic_option is True and method is "correlation_com", else None.

    Note:
        The function assumes that the number of rows in `vpath_data` and `y_data` are equal

    Example:
        vpath_data = np.random.randint(1, 4, size=(100, 5))  # Simulated hidden state path data
        y_data = np.random.rand(100, 3)  # Simulated dependent variable data
        n_states = 5
        pval, corr_coef, test_statistic_list, pval_list = within_session_continuous_test(vpath_data, y_data, n_states,
                                                                                          method="correlation_com",
                                                                                          Nperm=1000,
                                                                                          test_statistic_option=True)
        print("Correlation Coefficients:", corr_coef)
        print("P-values:", pval)
        print("Test Statistics:", test_statistic_list)
        print("Permutation P-values:", pval_list)
    """
    # Check validity of method
    valid_methods = ["regression", "correlation", "correlation_com"]
    check_value_error(method in valid_methods, "Invalid option specified for 'method'. Must be one of: " + ', '.join(valid_methods))

    # Get input shape information
    n_timepoints, n_ST, n_features, vpath_data, y_data = get_input_shape(vpath_data, y_data)
    n_predictions = y_data.shape[-1]

    # Initialize arrays based on shape of data shape and defined options
    pval, corr_coef, test_statistic_list, pval_list = initialize_arrays(vpath_data, n_features, n_predictions,
                                                                         n_timepoints, method, Nperm,
                                                                         test_statistic_option)

    # Print tqdm over n_timepoints if there are more than one timepoint
    for t in tqdm(range(n_timepoints)) if n_timepoints > 1 else range(n_timepoints):
        # Create test_statistic and pval_perms based on method
        test_statistic, pval_perms, proj = initialize_permutation_matrices(method, Nperm, n_features, n_predictions,
                                                                           y_data[t, :])

        for perm in tqdm(range(Nperm)) if n_timepoints == 1 else range(n_timepoints):
            # Perform permutation on vpath
            vpath_surrogate = within_session_continuous_surrogate_state_time(vpath_data[t, :], n_states, perm)
            test_statistic, pval_perms = test_statistic_calculations(vpath_surrogate, y_data[t, :], perm, pval_perms,
                                                                     test_statistic, proj, method)

        pval, corr_coef = get_pval(test_statistic, pval_perms, Nperm, method, t, pval, corr_coef)
        if test_statistic_option:
            test_statistic_list[t, :] = test_statistic
            # If pval_perms is empty (evaluates to False), the right-hand side of the assignment will be pval_list[t, :]
            # itself, meaning that the array will remain unchanged.
            pval_list[t, :] = pval_perms if np.any(pval_perms) else pval_list[t, :]
    pval =np.squeeze(pval)
    corr_coef =np.squeeze(corr_coef)   
    test_statistic_list =np.squeeze(test_statistic_list)   
    pval_list =np.squeeze(pval_list)    
    
    # Return values if test_statistic_option is True
    if test_statistic_option and method == "regression":
        return pval, test_statistic_list
    elif test_statistic_option and method == "correlation":
        return corr_coef, test_statistic_list
    elif test_statistic_option and method == "correlation_com":
        return pval, corr_coef, test_statistic_list, pval_list
    elif method == "regression":
        return pval
    elif method == "correlation":
        return corr_coef
    elif method == "correlation_com":
        return pval, corr_coef



def check_value_error(condition, error_message):
    """
    Checks a given condition and raises a ValueError with the specified error message if the condition is not met.

    Parameters:
    --------------
        condition (bool): The condition to check.
        error_message (str): The error message to raise if the condition is not met.
    """
    # Check if a condition is False and raise a ValueError with the given error message
    if not condition:
        raise ValueError(error_message)


def get_input_shape(X_data, y_data):
    """
    Computes the input shape parameters for permutation testing.

    Parameters:
    --------------
        X_data (numpy.ndarray): The input data array.
        y_data (numpy.ndarray): The dependent variable.

    Returns:
    ----------  
        n_timepoints (int): The number of timepoints.
        n_ST (int): The number of subjects/trials.
        n_features (int): The number of features.
        X_data (numpy.ndarray): The updated input data array.
        y_data (numpy.ndarray): The updated dependent variable.
    """
    # Get the input shape of the data and perform necessary expansions if needed
    if y_data.ndim == 1:
        y_data = np.expand_dims(y_data, axis=1)
        
    if len(X_data.shape) == 2:
        # Performing permutation testing for the whole data
        print("performing permutation testing for whole data")
        X_data = np.expand_dims(X_data, axis=0)
        y_data = np.expand_dims(y_data, axis=0)
        n_timepoints, n_ST, n_features = X_data.shape
        
    else:
        # Performing permutation testing per timepoint
        print("performing permutation testing per timepoint")
        n_timepoints, n_ST, n_features = X_data.shape
        

        # Tile the y_data if it doesn't match the number of timepoints in X_data
        if y_data.shape[0] != X_data.shape[0]:
            y_data = np.tile(y_data, (X_data.shape[0],1,1)) 
        
    return n_timepoints, n_ST, n_features, X_data, y_data


def initialize_permutation_matrices(method, Nperm, n_features, n_predictions, y_data):
    """
    Initializes the permutation matrices and projection matrix for permutation testing.

    Parameters:
    --------------
        method (str): The method to use for permutation testing.
        Nperm (int): The number of permutations.
        n_features (int): The number of features.
        n_predictions (int): The number of predictions.
        y_data (numpy.ndarray): The dependent variable.

    Returns:
    ----------  
        test_statistic (numpy.ndarray): The permutation array.
        pval_perms (numpy.ndarray): The p-value permutation array.
        proj (numpy.ndarray or None): The projection matrix (None for correlation methods).
    """
    # Initialize the permutation matrices based on the selected method
    if method in {"correlation", "correlation_com"}:
        test_statistic = np.zeros((Nperm, n_features, n_predictions))
        pval_perms = np.zeros((Nperm, n_features, n_predictions))
        proj = None
    else:
        test_statistic = np.zeros((Nperm, n_features))
        pval_perms = []
        regularization = 0.001
        proj = np.linalg.inv(y_data.T.dot(y_data) + regularization * np.eye(y_data.shape[1])).dot(y_data.T)
    return test_statistic, pval_perms, proj


def initialize_arrays(X_data, n_features, n_predictions, n_timepoints, method, Nperm, test_statistic_option):
    """
    Initializes the result arrays for permutation testing.

    Parameters:
    --------------
        X_data (numpy.ndarray): The input data array.
        n_features (int): The number of features.
        n_predictions (int): The number of predictions.
        n_timepoints (int): The number of timepoints.
        method (str): The method to use for permutation testing.
        Nperm (int): Number of permutations.
        test_statistic_option (bool): If True, return the test statistic values.

    Returns:
    ----------  
        pval (numpy array): p-values for the test (n_timepoints, n_features) if test_statistic_option is False, else None.
        corr_coef (numpy array): Correlation coefficient for the test (n_timepoints, n_features, n_predictions) if method=correlation or method = correlation_com, else None.
        test_statistic_list (numpy array): Test statistic values (n_timepoints, Nperm, n_features) or (n_timepoints, Nperm, n_features, n_predictions) if method=correlation or method = correlation_com, else None.
        pval_list (numpy array): P-values for each time point (n_timepoints, Nperm, n_features) or (n_timepoints, Nperm, n_features, n_predictions) if test_statistic_option is True and method is "correlation_com", else None.
    """

    # Initialize the arrays based on the selected method and data dimensions
    if len(X_data.shape) == 2:
        pval = np.zeros((n_timepoints, n_features))
        corr_coef = np.zeros((n_timepoints, n_features, n_predictions))
        if test_statistic_option==True:
            test_statistic_list = np.zeros((n_timepoints, Nperm, n_features))
            pval_list = np.zeros((n_timepoints, Nperm, n_features))
        else:
            test_statistic_list= None
            pval_list =None
    elif method == "correlation_com" or method == "correlation" :
        pval = np.zeros((n_timepoints, n_features, n_predictions))
        corr_coef = pval.copy()
        
        if test_statistic_option==True:    
            test_statistic_list = np.zeros((n_timepoints, Nperm, n_features, n_predictions))
            pval_list = np.zeros((n_timepoints, Nperm, n_features, n_predictions))
        else:
            test_statistic_list= None
            pval_list =None
        
    else:
        pval = np.zeros((n_timepoints, n_features))
        corr_coef = np.zeros((n_timepoints, n_features, n_predictions))
        if test_statistic_option==True:
            test_statistic_list = np.zeros((n_timepoints, Nperm, n_features))
            pval_list = np.zeros((n_timepoints, Nperm, n_features))
        else:
            test_statistic_list= None
            pval_list =None
            
    if method not in {"regression", "correlation", "correlation_com"}:
        raise ValueError("Invalid option specified. Must be 'regression', 'correlation', or 'correlation_com'.")
    return pval, corr_coef, test_statistic_list, pval_list


def calculate_X_t(X_data, confounds=None):
    """
    Calculate the X_t array for permutation testing.

    Parameters:
    --------------
        X_data (numpy.ndarray): The input data array.
        confounds (numpy.ndarray or None): The confounds array (default: None).

    Returns:
    ----------  
        numpy.ndarray: Calculated X_t array.
    """
    # Calculate the centered data matrix based on confounds (if provided)
    if confounds is not None:
        confounds -= np.mean(confounds, axis=0)
        X_t = X_data - np.dot(confounds, np.linalg.pinv(confounds).dot(X_data))
    else:
        X_t = X_data - np.mean(X_data, axis=0)
    return X_t


def between_subject_indices(Nperm, X_t, indices=False, exchangeable=True):
    """
    Generates between-subject indices for permutation testing.

    Parameters:
    --------------
        Nperm (int): The number of permutations.
        X_t (numpy.ndarray): The preprocessed data array.
        indices (bool): Flag indicating whether to use specific indices (default: False).
        exchangeable (bool): Flag indicating whether to use exchangeable permutations (default: False).

    Returns:
    ----------  
        permute_idx_list (numpy.ndarray): The between-subject indices array.
    """
    permute_idx_list = np.zeros((Nperm, X_t.shape[0]), dtype=int)
    for perm in range(Nperm):
        if perm == 0:
            permute_idx_list[perm] = np.arange(X_t.shape[0])
        elif perm > 0 and exchangeable==False:
            for t in range(np.diff(indices)[0]):
                idx_t = [i + t for i in indices]
                idx_t_perm = np.random.permutation(idx_t)
                for i in range(len(idx_t_perm)):
                    permute_idx_list[perm, :][idx_t[i]] = idx_t_perm[i]
        else:
            permute_idx_list[perm] = np.random.permutation(X_t.shape[0])
    return permute_idx_list

def get_pval(test_statistic, pval_perms, Nperm, method, t, pval, corr_coef):
    """
    Computes p-values and correlation matrix for permutation testing.

    Parameters:
    --------------
        test_statistic (numpy.ndarray): The permutation array.
        pval_perms (numpy.ndarray): The p-value permutation array.
        Nperm (int): The number of permutations.
        method (str): The method used for permutation testing.
        t (int): The timepoint index.
        pval (numpy.ndarray): The p-value array.
        corr_coef (numpy.ndarray): The correlation p-value array.

    Returns:
    ----------  
        
        pval (numpy.ndarray): Updated updated p-value .
        corr_coef (numpy.ndarray): Updated correlation p-value arrays.
    """
    if method == "regression":
        pval[t, :] = np.sum(test_statistic <= test_statistic[0], axis=0) / (Nperm + 1)
    elif method == "correlation":
        corr_coef[t, :] = np.sum(test_statistic <= test_statistic[0], axis=0) / (Nperm + 1)
    elif method == "correlation_com":
        corr_coef[t, :] = np.sum(test_statistic <= test_statistic[0], axis=0) / (Nperm + 1)
        pval[t, :] = np.sum(pval_perms <= pval_perms[0], axis=0) / (Nperm + 1)

    return pval, corr_coef


def get_indices_array(idx_data):
    """
    Generates an indices array based on given data indices.

    Parameters:
    --------------
        idx_data (numpy.ndarray): The data indices array.

    Returns:
    ----------  
        idx_array (numpy.ndarray): The generated indices array.
    """
    # Get an array of indices based on the given idx_data ranges
    max_value = np.max(idx_data[:, 1])
    idx_array = np.zeros(max_value + 1, dtype=int)
    for count, (start, end) in enumerate(idx_data):
        idx_array[start:end + 1] = count
    return idx_array


def within_session_between_trial(X_t, idx_array, perm):
    """
    Generates within-session between-trial data based on given indices.

    Parameters:
    --------------
        X_t (numpy.ndarray): The preprocessed data array.
        idx_array (numpy.ndarray): The indices array.
        perm (int): The permutation index.

    Returns:
    ----------  
        Xin (numpy.ndarray): The within-session between-trial data array.
    """
    # Perform within-session between-trial permutation based on the given indices
    if perm == 0:
        Xin = X_t
    else:
        Xin = np.zeros(X_t.shape)
        unique_indices = np.unique(idx_array)
        for i in range(unique_indices.size):
            X_index_subject = X_t[idx_array == unique_indices[i], :]
            X_perm = np.random.permutation(X_index_subject.shape[0])
            Xin[idx_array == unique_indices[i], :] = X_index_subject[X_perm, :]
    return Xin


def within_session_continuous_surrogate_state_time(viterbi_path, n_states, perm):
    """
    Generates a surrogate state-time matrix based on a given Viterbi path.

    Parameters:
    --------------
    viterbi_path (numpy.ndarray): 1D array or 2D matrix containing the Viterbi path.
    n_states (int): Number of states in the hidden Markov model.
    perm (int): Number of permutations to generate surrogate state-time matrices.

    Returns:
    ----------  
    numpy.ndarray: Surrogate state-time matrix as a 2D matrix representing the Viterbi path in each row.
    """
    if perm == 0:
        if len(viterbi_path.shape) == 1:
            vpath_surrogate = viterbi_path_to_stc(viterbi_path, n_states)
        else:
            vpath_surrogate = viterbi_path.copy().astype(int)
    else:
        vpath_surrogate = surrogate_viterbi_path(viterbi_path, n_states)

    return vpath_surrogate

def viterbi_path_to_stc(viterbi_path, n_states):
    """
    Convert Viterbi path to state-time matrix.

    Parameters:
    --------------
    viterbi_path (numpy.ndarray): 1D array or 2D matrix containing the Viterbi path.
    n_states (int): Number of states in the hidden Markov model.

    Returns:
    ----------  
    numpy.ndarray: State-time matrix where each row represents a time point and each column represents a state.
    """
    stc = np.zeros((len(viterbi_path), n_states), dtype=int)
    stc[np.arange(len(viterbi_path)), viterbi_path] = 1
    return stc


def surrogate_viterbi_path(viterbi_path, n_states):
    """
    Generate surrogate Viterbi path based on state-time matrix.

    Parameters:
    --------------
    viterbi_path (numpy.ndarray): 1D array or 2D matrix containing the Viterbi path.
    n_states (int): Number of states in the hidden Markov model.

    Returns:
    ----------  
    numpy.ndarray: Surrogate Viterbi path as a 1D array representing the state indices.
    """
    if len(viterbi_path.shape) == 1:
        viterbi_path_1D = viterbi_path.copy()
        stc = viterbi_path_to_stc(viterbi_path, n_states)
    else:
        viterbi_path_1D = np.argmax(viterbi_path, axis=1)
        stc = viterbi_path.copy()

    state_probs = stc.mean(axis=0).cumsum()
    viterbi_path_surrogate = np.zeros(viterbi_path_1D.shape)
    index = 0

    while index < len(viterbi_path_1D):
        t_next = np.where(viterbi_path_1D[index:] != viterbi_path_1D[index])[0]

        if len(t_next) == 0:
            t_next = len(viterbi_path_1D)
        else:
            t_next = t_next[0]
            t_next = index + t_next

        state = np.where(state_probs >= random.uniform(0, 1))[0][0]
        state += 1

        viterbi_path_surrogate[index:t_next] = state
        index = t_next

    vpath_surrogate = np.zeros_like(stc).astype(int)
    vpath_surrogate[np.arange(len(viterbi_path_surrogate)), viterbi_path_surrogate.astype(int) - 1] = 1
    return vpath_surrogate


def test_statistic_calculations(Xin, y_data, perm, pval_perms, test_statistic, proj, method):
    """
    Calculates the test_statistic array and pval_perms array based on the given data and method.

    Parameters:
    --------------
        Xin (numpy.ndarray): The data array.
        y_data (numpy.ndarray): The dependent variable.
        perm (int): The permutation index.
        pval_perms (numpy.ndarray): The p-value permutation array.
        test_statistic (numpy.ndarray): The permutation array.
        proj (numpy.ndarray or None): The projection matrix (None for correlation methods).
        method (str): The method used for permutation testing.

    Returns:
    ----------  
        test_statistic (numpy.ndarray): Updated test_statistic array.
        pval_perms (numpy.ndarray): Updated pval_perms array.
    """
    if method == 'regression':
        beta = np.dot(proj, Xin)
        test_statistic[perm,:] = np.sqrt(np.sum((y_data.dot(beta) - Xin) ** 2, axis=0))
    elif method == 'correlation':
        corr_coef = np.corrcoef(Xin, y_data, rowvar=False)
        corr_matrix = corr_coef[:Xin.shape[1], Xin.shape[1]:]
        test_statistic[perm, :, :] = np.abs(corr_matrix)
    elif method == "correlation_com":
        corr_coef = np.corrcoef(Xin, y_data, rowvar=False)
        corr_matrix = corr_coef[:Xin.shape[1], Xin.shape[1]:]
        pval_matrix = np.zeros(corr_matrix.shape)
        for i in range(Xin.shape[1]):
            for j in range(y_data.shape[1]):
                _, pval_matrix[i, j] = pearsonr(Xin[:, i], y_data[:, j])

        test_statistic[perm, :, :] = np.abs(corr_matrix)
        pval_perms[perm, :, :] = pval_matrix
    return test_statistic, pval_perms
