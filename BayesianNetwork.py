import numpy as np
import os
import sys

# Define the variables and values
variables = np.array(['OC','Trav','Fraud','FP','IP','CRP'])
values = np.array(['false','true'])

# restrict function
def restrict(factor, variable, value):
    """
    Restricts a factor to a given value on a given variable.
    
    Parameters:
    factor: np.ndarray
    The factor to restrict.
    variable: int
    The index of the variable to restrict.
    value: int
    The value to restrict the variable to.
    
    Returns:
    np.ndarray
    The restricted factor.
    """
    
    newShape = np.array(factor.shape)
    newShape[variable] = 1
    sliceList = [slice(None)] * factor.ndim
    sliceList[variable] = value
    
    return factor[tuple(sliceList)].reshape(newShape)

# sumout function
def sumout(factor,variable):
    """
    Sums out a variable from a factor.
    
    Parameters:
    factor: np.ndarray
    The factor to sum out the variable from.
    variable: int
    The index of the variable to sum out.
    
    Returns:
    np.ndarray
    The factor with the variable summed out.
    """
    return np.sum(factor,axis=variable,keepdims=True)

# multiply function
def multiply(factor1,factor2):
    """
    Multiplies two factors together.

    Parameters:
    factor1: np.ndarray
    The first factor.
    factor2: np.ndarray
    The second factor.

    Returns:
    np.ndarray
    """
    return factor1*factor2

# normalize function
def normalize(factor):
    """
    Normalizes a factor.

    Parameters:
    factor: np.ndarray
    The factor to normalize.

    Returns:
    np.ndarray
    """

    return factor / np.sum(factor.flatten())

# inference function
def inference(factorList,queryVariables,orderedListOfHiddenVariables,evidenceList):
    """
    Perform inference on a Bayesian network.

    Parameters:
    factorList: list
    A list of factors representing the Bayesian network.
    queryVariables: list
    A list of variables to query.
    orderedListOfHiddenVariables: list
    A list of hidden variables in the network.
    evidenceList: list
    A list of evidence variables and their values.

    Returns:
    np.ndarray
    """

    # restrict factors
    for index in np.arange(len(factorList)):
        shape = np.array(factorList[index].shape)
        for evidence in evidenceList:
            if shape[evidence[0]] > 1:
                factorList[index] = restrict(factorList[index],evidence[0],evidence[1])
        shape = np.array(factorList[index].shape)

    # eliminate each hidden variable
    print ("Eliminating hidden variables\n")
    hiddenId = 5
    for variable in orderedListOfHiddenVariables:
        print("Eliminating {}".format(variables[variable]))

        # find factors that contain the variable to be eliminated
        factorsToBeMultiplied = []
        for index in np.arange(len(factorList)-1,-1,-1):
            shape = np.array(factorList[index].shape)
            if shape[variable] > 1:
                factorsToBeMultiplied.append(factorList.pop(index))

        # multiply factors
        product = factorsToBeMultiplied[0]
        for factor in factorsToBeMultiplied[1:]:
            product = multiply(product,factor)

        # sumout variable
        newFactor = sumout(product,variable)
        factorList.append(newFactor)
        shape = np.array(newFactor.shape)
        hiddenId = hiddenId + 1
        print("New factor: f{}({})={}\n".format(hiddenId,variables[shape>1],np.squeeze(newFactor)))

    # multiply remaining factors
    print ("Multiplying factors")
    answer = factorList[0]
    for factor in factorList[1:]:
        answer = multiply(answer,factor)
        shape = np.array(answer.shape)
        print("Answer pre-normalization: f{}({})={}\n".format(hiddenId+1,variables[shape>1],np.squeeze(answer)))

    # normalize answer
    print ("Normalizing ...")
    answer = normalize(answer)
    shape = np.array(answer.shape)
    print("Normalized answer: f{}({})={}\n".format(hiddenId+2,variables[shape>1],np.squeeze(answer)))
    return answer

def print_helper(query_variables, evidence_list, inference_result):
    """
    Print the result of the inference in a readable format.

    Parameters:
    query_variables: list
    A list of the query variables.


    evidence_list: list
    A list of evidence variables and their values.

    inference_result: np.ndarray
    The result of the inference.

    """

    print('Result:')
    print('P(', end='')
    
    # Print the query variables
    for i, query_variable in enumerate(query_variables):
        print(query_variable, end='' if i == len(query_variables) - 1 else ',')
    
    # Print the evidence list if it's not empty
    if evidence_list:
        print(' | ', end='')
        for evidence in evidence_list:
            variable, value = evidence
            # The value is expected to be a boolean, so print accordingly
            value_str = variable if value else f'~{variable}'
            print(value_str, end=',')
        print('\b', end='')  # Remove the last comma

    # Print the result
    print(') = {}'.format(np.squeeze(inference_result)))

# Main function
def main() -> None:

    print("Welcome to the Bayesian Network Inference Program\n")
    # Initialize the program
    Ans = 'Y'
    # Loop until the user wants to exit
    while (Ans == 'Y' or Ans == 'y'):

        # Define the variables  
        OC=0
        Trav=1
        Fraud=2
        FP=3
        IP=4
        CRP=5

        # Define the values
        false=0
        true=1

        # Define the factors

        # Pr(OC)
        # [OC=false, OC=true]
        f0 = np.array([0.2,0.8])
        f0 = f0.reshape(2,1,1,1,1,1)
        print ("Pr(OC)={}\n".format(np.squeeze(f0)))

        # Pr(Trav)
        # [Trav=false, Trav=true]
        f1 = np.array([0.95, 0.05])
        f1 = f1.reshape(1,2,1,1,1,1)
        print ("Pr(Trav)={}\n".format(np.squeeze(f1)))

        # Pr(Fraud|Trav)
        # [[Trav=false,Fraud=false],[Trav=false,Fraud=True],[Trav=true,Fraud=false],[Trav=true,Fraud=True]]
        f2 = np.array([[0.996,0.004],[0.99,0.01]])
        f2 = f2.reshape(1,2,2,1,1,1)
        print ("Pr(Fraud|Trav)={}\n".format(np.squeeze(f2)))

        # Pr(FP|Fraud,Trav)
        # [[[Fraud=false,Trav=false],FP=false], [[Fraud=false,Trav=false],FP=true], [[Fraud=false,Trav=true],FP=false], [[Fraud=false,Trav=true],FP=true], [[Fraud=true,Trav=false],FP=false], [[Fraud=true,Trav=false],FP=true], [[Fraud=true,Trav=true],FP=false], [[Fraud=true,Trav=true],FP=true]]
        f3 = np.array([[[0.99, 0.01],[0.9,0.1]],[[0.1,0.9],[0.1,0.9]]])
        f3 = f3.reshape(1,2,2,2,1,1)
        print ("Pr(FP|Fraud,Trav)={}\n".format(np.squeeze(f3)))

        # Pr(IP|OC,Fraud)
        # [[[OC=false,Fraud=false],IP=false], [[OC=false,Fraud=false],IP=true], [[OC=false,Fraud=true],IP=false], [[OC=false,Fraud=true],IP=true], [[OC=true,Fraud=false],IP=false], [[OC=true,Fraud=false],IP=true], [[OC=true,Fraud=true],IP=false], [[OC=true,Fraud=true],IP=true]]
        f4 = np.array([[[0.999, 0.001],[0.949,0.051]],[[0.9,0.1],[0.85,0.15]]])
        f4 = f4.reshape(2,1,2,1,2,1)
        print ("Pr(IP|OC,Fraud)={}\n".format(np.squeeze(f4)))

        # Pr(CRP|OC)
        # [[OC=false,CRP=false], [OC=false,CRP=true], [OC=true,CRP=false], [OC=true,CRP=true]]
        f5 = np.array([[0.99,0.01],[0.9,0.1]])
        f5 = f5.reshape(2,1,1,1,1,2)
        print ("Pr(CRP|OC)={}\n".format(np.squeeze(f5)))

        # Define the factor list
        factorList = [f0, f1, f2, f3, f4, f5]

        # Query
        case = input('Enter 1 for default query or 2 for custom query: ')

        # Check the input
        if case != '1' and case != '2':
            print('Invalid input. Exiting program...')
            sys.exit()
        
        if (case == '1'): #default options
            print('1. P(Fraud)\n2. P(Fraud|FP,~IP,CRP)\n3. P(Fraud|FP,~IP,CRP,Trav)\n4. P(Fraud|IP)\n5. P(Fraud|IP,CRP)')
            query_num = input('Enter query number: ')
            if query_num == '1':
                f6 = inference(factorList,Fraud,[Trav, FP, IP, OC, CRP],[])
                print_helper(['Fraud'], [], f6)
            elif query_num == '2':
                f7 = inference(factorList,Fraud,[Trav,OC],[[FP,true],[IP,false],[CRP,true]])
                print ("P(Fraud|FP,~IP,CRP)={}\n".format(np.squeeze(f7)))
            elif query_num == '3':
                f8 = inference(factorList,Fraud,[OC],[[FP,true],[IP,false],[CRP,true],[Trav,true]])
                print ("P(Fraud|FP,~IP,CRP,Trav)={}\n".format(np.squeeze(f8)))
            elif query_num == '4':
                f9 = inference(factorList,Fraud,[Trav,FP,OC,CRP],[[IP,true]])
                print ("P(Fraud|IP)={}\n".format(np.squeeze(f9)))
            elif query_num == '5':
                f10 = inference(factorList,Fraud,[Trav,FP,OC],[[IP,true],[CRP,true]])
                print ("P(Fraud|IP,CRP)={}\n".format(np.squeeze(f10)))
            else:
                print('Invalid input. Exiting program...')
                sys.exit()

        else: #custom query 
            query_variables = []
            query = input('Enter query variables separated by commas: ')
            query = query.split(',')
            for i in query:
                if i == 'OC':
                    query_variables.append(OC)
                elif i == 'Trav':
                    query_variables.append(Trav)
                elif i == 'Fraud':
                    query_variables.append(Fraud)
                elif i == 'FP':
                    query_variables.append(FP)
                elif i == 'IP':
                    query_variables.append(IP)
                elif i == 'CRP':
                    query_variables.append(CRP)
                else:
                    print('Invalid input. Exiting program...')
                    sys.exit()
            # Enter evidence
            evidence = input('Enter evidence variables separated by commas: ')
            if(evidence == ''):
                evidence_list = []
                hidden_variables = []
                for i in range(0, 6):
                    if i not in query_variables and i not in [evidence[0] for evidence in evidence_list]:
                        hidden_variables.append(i)
                inference_result = inference(factorList, query_variables, hidden_variables, [])
                print_helper(query, [], inference_result)
            else:
                print_evidence =[]
                evidence = evidence.split(',')
                evidence_list = []
                for i in range(0, len(evidence), 2):
                    variable = evidence[i]  # This will be 'OC', 'Trav', etc.
                    value_str = evidence[i+1]  # This will be 'true' or 'false'

                    # Check the variable name and append the corresponding object
                    if variable == 'OC':
                        evidence_var = OC
                    elif variable == 'Trav':
                        evidence_var = Trav
                    elif variable == 'Fraud':
                        evidence_var = Fraud
                    elif variable == 'FP':
                        evidence_var = FP
                    elif variable == 'IP':
                        evidence_var = IP
                    elif variable == 'CRP':
                        evidence_var = CRP
                    else:
                        print('Invalid input. Exiting program...')
                        sys.exit()
                    
                    if value_str.lower() == 'true':
                        evidence_value = true
                    elif value_str.lower() == 'false':
                        evidence_value = false
                    else:
                        print('Invalid input. Exiting program...')
                        sys.exit()
                    
                    # Append the variable-value pair to the evidence list
                    evidence_list.append([evidence_var, evidence_value])
                    print_evidence.append([variable, evidence_value])
                hidden_variables = []
                for i in range(0, 6):
                    if i not in query_variables and i not in [evidence[0] for evidence in evidence_list]:
                        hidden_variables.append(i)
                inference_result = inference(factorList, query_variables, hidden_variables, evidence_list)
                print_helper(query, print_evidence, inference_result)
            
        # Ask the user if they want to do another query
        Ans = input('Do you want to do another query? (Y/N): ')
        if Ans == 'N' or Ans == 'n':
            print('Thank you for using the program. Goodbye!') # Exit the program
            sys.exit()
        if Ans != 'Y' and Ans != 'y':
            print('Invalid input. Exiting program...') 
            sys.exit()
        # Clear the screen
        os.system('cls||clear')

if __name__ == '__main__':
    main()