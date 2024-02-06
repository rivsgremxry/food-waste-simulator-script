# Import necessary libraries and modules
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Set the title of the Streamlit web app
st.write("""
# Food waste simulation App
""")

# Set up sidebar for user-defined breakfast parameters
st.sidebar.header('Breakfast parameters')

########################################################################
#             User defined breakfast parameters of experiment          #
########################################################################

def breakfast_parameters():
    # Generations per one case
    number_of_examples = st.sidebar.slider('Number of examples', 1, 10000, 1)
    # Units are minutes
    min_breakfast_duration = st.sidebar.slider('Breakfast duration (min)', 1, 5, 5)
    max_breakfast_duration = st.sidebar.slider('Breakfast duration (max)', 1, 20, 20)
    breakfast_duration_step = st.sidebar.slider('Breakfast duration step', 0.1, 1.0, 1.0)
    data = {'number_of_examples': number_of_examples,
            'min_breakfast_duration': min_breakfast_duration,
            'max_breakfast_duration': max_breakfast_duration,
            'breakfast_duration_step': breakfast_duration_step}
    param = pd.DataFrame(data, index=[0])
    return param

df_breakfast = breakfast_parameters()

########################################################################
#             User defined dislike parameters of experiment            #
########################################################################

# Units are percents
min_dislike_probability = 0.00
max_dislike_probability = 0.25
dislike_probability_step = 0.01

########################################################################
#                     Matrix of simulation results                     #
########################################################################

# Vertical dimension will be related with breakfast duration
max_breakfast_duration = df_breakfast.max_breakfast_duration
xLength = (int)((max_breakfast_duration - df_breakfast.min_breakfast_duration) / df_breakfast.breakfast_duration_step)

# Horizontal dimension will be related with dislike probability
max_dislike_probability = max_dislike_probability
yLength = (int)((max_dislike_probability - min_dislike_probability) / dislike_probability_step)

results = np.empty((xLength+1, yLength+1))

st.write("Size of result matrix: " + str(results.shape))

########################################################################
#                              Meal type                               #
########################################################################

# "SOUP_AND_MAIN": "main" means a hardly chewing main course

def GenerateMealType():
    mealType = np.random.choice(["EASILY_CHEWING_MAIN", "HARDLY_CHEWING_MAIN", "SOUP_AND_MAIN"])
    return mealType

# Soup parameters
SOUP_MEAN = 225
SOUP_STDDEV = 30

# Hardly chewing main course parameters
EMAIN_MEAN = 285
EMAIN_STDDEV = 30

# Hardly chewing main course parameters
HMAIN_MEAN = 660
HMAIN_STDDEV = 120

# Global minimum - one minute
GLOBAL_MIN = 60

def GenerateRequiredMealDuration( mealType ):
    if mealType == "HARDLY_CHEWING_MAIN":
        length = np.random.normal( HMAIN_MEAN, HMAIN_STDDEV )
    elif mealType == "EASILY_CHEWING_MAIN":
        length = np.random.normal( EMAIN_MEAN, EMAIN_STDDEV )
    else: # Soup
        length = np.random.normal( SOUP_MEAN, SOUP_STDDEV )

    if length < GLOBAL_MIN:
        return GLOBAL_MIN
    else:
        return length


st.write("""
### Test normal distribution
""")

########################################################################
#                       Test normal distribution                       #
########################################################################

soup = np.empty(10000)
eMain = np.empty(10000)
hMain = np.empty(10000)


for i in range(10000):
    soup[i] = GenerateRequiredMealDuration( "SOUP" )
    eMain[i] = GenerateRequiredMealDuration( "EASILY_CHEWING_MAIN" )
    hMain[i] = GenerateRequiredMealDuration( "HARDLY_CHEWING_MAIN" )

fig, axs = plt.subplots(1,3,figsize=(18,6))

plt.subplot( 1, 3, 1 )
plt.title("Soup")
counts, bins = np.histogram(soup)
plt.hist(bins[:-1], bins, weights=counts)

plt.subplot( 1, 3, 2 )
plt.title("Easily Main")
counts, bins = np.histogram(eMain)
plt.hist(bins[:-1], bins, weights=counts)

plt.subplot( 1, 3, 3 )
plt.title("Hardly Main")
counts, bins = np.histogram(hMain)
plt.hist(bins[:-1], bins, weights=counts)

st.pyplot(fig)

########################################################################
#                      Generate Dislike Parameters                     #
########################################################################

# Units are percents
LEFT_BEHIND_MEAN = 0.15
LEFT_BEHIND_STDDEV = 0.025

def GenerateLeftBehind( dislikeProbab ):
    isFavoriteDish = np.random.choice([True, False], p=[1.0-dislikeProbab, dislikeProbab])
    if isFavoriteDish:
        return 0.0
    else:
        return np.random.normal( LEFT_BEHIND_MEAN, LEFT_BEHIND_STDDEV )


st.write("""
### Test dislike function
""")

########################################################################
#                        Test Dislike Function                         #
########################################################################

leftBehind0 = np.empty(10000)
leftBehind10 = np.empty(10000)
leftBehind20 = np.empty(10000)
leftBehind30 = np.empty(10000)

for i in range(10000):
    leftBehind0[i] = GenerateLeftBehind( 0.0 )
    leftBehind10[i] = GenerateLeftBehind( 0.1 )
    leftBehind20[i] = GenerateLeftBehind( 0.2 )
    leftBehind30[i] = GenerateLeftBehind( 0.3 )

fig, axs = plt.subplots(1,4,figsize=(18,6))

plt.subplot( 1, 4, 1 )
plt.title("Dislike 0%")
counts, bins = np.histogram(leftBehind0)
plt.hist(bins[:-1], bins, weights=counts)


plt.subplot( 1, 4, 2 )
plt.title("Dislike 10%")
counts, bins = np.histogram(leftBehind10)
plt.hist(bins[:-1], bins, weights=counts)


plt.subplot( 1, 4, 3 )
plt.title("Dislike 20%")
counts, bins = np.histogram(leftBehind20)
plt.hist(bins[:-1], bins, weights=counts)

plt.subplot( 1, 4, 4 )
plt.title("Dislike 30%")
counts, bins = np.histogram(leftBehind30)
plt.hist(bins[:-1], bins, weights=counts)

st.pyplot(fig)

########################################################################
#                      Waste Calculation Functions                     #
########################################################################

def CalcBiasTime(requiredMealDuration):
    return (2.0 / 5.0) * requiredMealDuration

def CalcStartVelocity(requiredMealDuration):
    return 5.0 / (3.0 * requiredMealDuration)

def CalcFirstPart( requiredMealDuration, startVelocity, mealDuration ):
    return ((startVelocity * mealDuration) - ((5.0 / 8.0) * startVelocity * math.pow(mealDuration, 2.0) / requiredMealDuration))

def CalcSecondPart( startVelocity, mealDuration ):
    return startVelocity * mealDuration / 2.0

def CalcSecondPartWithDislike( startVelocity, mealDuration, leftBehind ):
    if leftBehind > 0.5:
        leftBehind = 0.5
    mealDuration = mealDuration - (2.0 * leftBehind / startVelocity)
    return CalcSecondPart( startVelocity, mealDuration )

# Functions for simulation
def CalculateFoodWaste( mealType, dislikeProbab, breakfastDuration ):

    # In this experiment, it is considered similar for all types
    leftBehind = GenerateLeftBehind( dislikeProbab )

    if mealType == "EASILY_CHEWING_MAIN":

        # Required time to eat whole dish
        # The whole dish is represented as 1.0 (100%)
        requiredMealDuration = GenerateRequiredMealDuration( "EASILY_CHEWING_MAIN" )

        mealDuration = (1.0 - leftBehind) * requiredMealDuration
        if mealDuration > breakfastDuration:
            mealDuration = breakfastDuration

        eattenPart = mealDuration / requiredMealDuration

        return (1.0 - eattenPart)

    elif mealType == "SOUP":

        # Required time to eat whole dish
        # The whole dish is represented as 1.0 (100%)
        requiredMealDuration = GenerateRequiredMealDuration( "SOUP" )

        mealDuration = (1.0 - leftBehind) * requiredMealDuration
        if mealDuration > breakfastDuration:
            mealDuration = breakfastDuration

        eattenPart = mealDuration / requiredMealDuration

        return (1.0 - eattenPart)

    elif mealType == "HARDLY_CHEWING_MAIN":

        # Required time to eat whole dish
        # The whole dish is represented as 1.0 (100%)
        requiredMealDuration = GenerateRequiredMealDuration( "HARDLY_CHEWING_MAIN" )

        startOfSecondPart = CalcBiasTime(requiredMealDuration)
        startVelocity = CalcStartVelocity(requiredMealDuration)

        if startOfSecondPart >= breakfastDuration:
            mealDuration = breakfastDuration
            eattenPart = CalcFirstPart( requiredMealDuration, startVelocity, mealDuration )
            return (1.0 - eattenPart)
        else:
            mealDuration = startOfSecondPart
            eattenPart1 = CalcFirstPart( requiredMealDuration, startVelocity, mealDuration )

            mealDuration = requiredMealDuration - (2.0 * leftBehind / startVelocity)

            if mealDuration > breakfastDuration:
                mealDuration = breakfastDuration

            mealDuration = mealDuration - startOfSecondPart

            eattenPart2 = startVelocity * mealDuration / 2.0

            eattenPart = 1.0 - round(eattenPart1 + eattenPart2, 5)

            return eattenPart

    else:
        # Required time to eat whole dish
        # The whole dish is represented as 2.0 (200%), because it contains two dishes

        soupWaste = CalculateFoodWaste( "SOUP", dislikeProbab, breakfastDuration )
        mainDishWaste = CalculateFoodWaste( "HARDLY_CHEWING_MAIN", dislikeProbab, breakfastDuration )

        return (soupWaste + mainDishWaste) / 2.0


########################################################################
#                             Simulation                               #
########################################################################

st.write("""
### Simulation
""")

# Check if the user wants to show simulation data
if st.checkbox('Show Simulation data'):


    startX = float ( df_breakfast.min_breakfast_duration )
    stopY = float ( df_breakfast.max_breakfast_duration + df_breakfast.breakfast_duration_step )
    stepZ = float ( df_breakfast.breakfast_duration_step )
    number_of_examples = int ( df_breakfast.number_of_examples )
    row = 0

    # Change of breakfast duration
    for breakfastDuration in np.arange( startX, stopY, stepZ ):

        # minutes are converted to seconds
        breakfastDuration = breakfastDuration * 60.0

        # Y dimension
        column = 0

        # Change of dislike probability
        for dislikeProbab in np.arange( min_dislike_probability, max_dislike_probability+dislike_probability_step, dislike_probability_step ):

            producedFood = 0.0
            producedWaste = 0.0

            # Generation of examples
            for k in range( number_of_examples ):

                mealType = GenerateMealType()

                if mealType == "SOUP_AND_MAIN":
                    producedFood = producedFood + 2.0
                else:
                    producedFood = producedFood + 1.0

                producedWaste = producedWaste + CalculateFoodWaste( mealType, dislikeProbab, breakfastDuration )

            if producedWaste > 0.0:
                results[row, column] = producedWaste / number_of_examples
            else:
                results[row, column] = 0.0

            column = column + 1

        row = row + 1

    st.write("Calculated table: rows - breakfast duration, columns - dislike probability")
    st.write(results)

########################################################################################################

    from mpl_toolkits import mplot3d

    x_range = np.arange( startX, stopY, stepZ )
    y_range = np.arange( min_dislike_probability, max_dislike_probability+dislike_probability_step, dislike_probability_step )

    # Breakfast duration
    x = np.outer(x_range, np.ones(len(y_range)))
    # Dislike probability
    y = np.outer(np.ones(len(x_range)), y_range) * 100
    # Food waste
    z = results * 100

    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')

    ax.plot_surface(x, y, z,cmap='cividis', edgecolor='black')

    ax.set_xlabel('Breakfast duration (min)', fontweight ='bold', fontsize = 12)
    ax.set_ylabel('Dislike probability (%)', fontweight ='bold', fontsize = 12)
    ax.set_zlabel('Food waste (%)', fontweight ='bold', fontsize = 12)

    ax.set_title('Experiment results', fontweight ='bold', fontsize = 12)

    st.pyplot(fig)


#######################################################################################

    fig = plt.figure(figsize=(10,5))
    ax = plt.axes()

    # Breakfast duration Min and Max
    ax = plt.subplot(1, 2, 1)
    ax.set_xlabel('Dislike probability (%)', fontweight ='bold', fontsize = 12)
    ax.set_ylabel('Food waste (%)', fontweight ='bold', fontsize = 12)

    plt.plot( y[0,:], z[0,:], linestyle='--', label='Breakfast duration = ' + str(float(df_breakfast.min_breakfast_duration)) + ' min', color='black' )
    plt.plot( y[0,:], z[len(z)-1,:], linestyle=':', label='Breakfast duration = ' + str(float(df_breakfast.max_breakfast_duration)) + ' min', color='black')
    ax.legend(loc="right", fontsize=12)

    plt.grid(True, linewidth=0.5, color='#cccccc', linestyle='-')

    # Dislike probability Min and Max
    ax = plt.subplot(1, 2, 2)
    ax.set_xlabel('Breakfast duration (min)', fontweight ='bold', fontsize = 12)
    ax.set_ylabel('Food waste (%)', fontweight ='bold', fontsize = 12)

    plt.plot( x[:,0], z[:,0], linestyle='--', label='Dislike probability = ' + str(min_dislike_probability*100), color='black' )
    plt.plot( x[:,0], z[:,len(z)-1], linestyle=':', label='Dislike probability = ' + str(max_dislike_probability*100), color='black' )
    ax.legend(loc="upper right", fontsize=12)

    plt.grid(True, linewidth=0.5, color='#cccccc', linestyle='-')

    st.pyplot(fig)
