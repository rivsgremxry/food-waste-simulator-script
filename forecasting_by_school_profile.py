# Import necessary libraries and modules
import streamlit as st
import pandas as pd
import numpy as np
import math

# Set the title of the Streamlit web app
st.title( 'E-Mentors: Pārtikas atkritumu prognozēšana pēc skolas profila' )

########################################################################
#                        Simulation parameters                         #
########################################################################

# How many time simulation is launched
number_of_examples = 10000

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

########################################################################
#                          Start User Panel                            #
########################################################################

st.sidebar.header('Skolas parametri')

########################################################################
#             User defined lunch parameters of experiment              #
########################################################################

def lunch_parameters():
    # Units are minutes
    lunch_duration = st.sidebar.slider( 'Pusdienu ilgums (minūtes):', min_value=1, max_value=60, value=15, step=1 )
    data = {'lunch_duration': lunch_duration}
    param = pd.DataFrame( data, index=[0] )
    return param

df_lunch = lunch_parameters()

########################################################################
#             User defined dislike parameters of experiment            #
########################################################################

def dislike_parameters():
    # Units are percents
    dislike_probability = st.sidebar.slider( 'Cik bērniem nepatīk ēdiens (%):', min_value=0, max_value=100, value=25, step=1 )
    data = {'dislike_probability': dislike_probability}
    param = pd.DataFrame(data, index=[0])
    return param

df_dislike = dislike_parameters()

########################################################################
#            User defined meal type parameters of experiment           #
########################################################################

def mealType_parameters():

    # Units are percents
    eMean = st.sidebar.slider( 'Kārstās uzkodas (%):', 0, 100, 33, 1 )
    hMean = st.sidebar.slider( 'Pamatēdiens (%):', 0, 100, 33, 1 )
   
    soupAndMain = 100 - eMean - hMean
    st.sidebar.write('Zupa un pamatēdiens (%): ', soupAndMain)

    if( soupAndMain < 0 ):
        st.error( 'Kopīgais porciju veidu skaits nav vienāds ar 100%' )
        st.stop()

    data = { 'eMean': eMean, 'hMean': hMean }
    param = pd.DataFrame(data, index=[0])
    return param

df_mealType = mealType_parameters()

########################################################################
#                              Meal type                               #
########################################################################

# "SOUP_AND_MAIN": "main" means a hardly chewing main course

eMean = float ( ( df_mealType.eMean ) / 100.0 )
hMean = float ( ( df_mealType.hMean ) / 100.0 )
soupAndMain = float ( 1.0 - eMean - hMean )

# Function to generate meal type
def GenerateMealType():
    mealType = np.random.choice(["EASILY_CHEWING_MAIN", "HARDLY_CHEWING_MAIN", "SOUP_AND_MAIN"], p=[eMean, hMean, soupAndMain])
    return mealType

# Function to generate required meal duration
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

########################################################################
#                      Generate Dislike Parameters                     #
########################################################################

# Function to generate dislike parameters
# Units are percents
LEFT_BEHIND_MEAN = 0.15
LEFT_BEHIND_STDDEV = 0.025

def GenerateLeftBehind( dislikeProbab ):
    isFavoriteDish = np.random.choice([True, False], p=[1.0-dislikeProbab, dislikeProbab])
    if isFavoriteDish:
        return 0.0
    else:
        return np.random.normal( LEFT_BEHIND_MEAN, LEFT_BEHIND_STDDEV )

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
#                         Start Simulation                             #
########################################################################

breakfastDuration = float ( df_lunch.lunch_duration ) * 60.0
dislikeProbab = float ( df_dislike.dislike_probability ) / 100.0

producedFood = 0.0      # Number of portions
producedWaste = 0.0     # Uneatten parts of portions

# Generation of examples
for k in range( number_of_examples ):

    mealType = GenerateMealType()

    if mealType == "SOUP_AND_MAIN":
        producedFood = producedFood + 2.0
    else:
        producedFood = producedFood + 1.0

    producedWaste = producedWaste + CalculateFoodWaste( mealType, dislikeProbab, breakfastDuration )

result = ( producedWaste / producedFood ) * 100.0

########################################################################
#                           End Simulation                             #
########################################################################

st.subheader('Pārtikas atkritumi šogad bija, apmēram:')
str = str( round(result, 2) ) + '%'
st.text( str )
