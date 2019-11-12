# Load all relevant packages for this notebook:
import numpy as np
import pandas as pd
from sklearn import utils, model_selection, metrics, linear_model, neighbors, ensemble
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal, stats
from IPython.display import display
from math import ceil

def getWeatherData(cities, extractRain = True, deriv = False):
    '''This function takes a list of cites as the input argument and extracts 
    the temporal weather data that we want to start exploring the data. Weather 
    descriptions are classified into either rain or not. The numerical variables 
    are z-scored to allow for comparisons between different variables that have 
    different scales and mean values.
    
    Functionality was also added to extract derivaties of numerical factors

    e.g.: 
    weatherData = getWeatherData(cities = ['NewYork'])
    '''
    
    # First, determine cities' indices in the dataset (they are consistent throughout)
    df = pd.read_csv('historical-hourly-weather-data/city_attributes.csv',usecols =[0])
    # Concatenate cities that are open compound words (e.g. New York or San Francisco):
    df.City = df.City.str.strip().str.replace(' ', '')
    allCities = list(df.City);
    cityIdxs=[]
    for city in cities:
        cityIdxs.append(allCities.index(city)+1)

    weatherFiles = ['historical-hourly-weather-data/humidity.csv',
    'historical-hourly-weather-data/wind_direction.csv',
    'historical-hourly-weather-data/temperature.csv',
    'historical-hourly-weather-data/pressure.csv',
    'historical-hourly-weather-data/wind_speed.csv',
    'historical-hourly-weather-data/weather_description.csv']

    # Initiate data structure with timestamps: 
    weatherData = [pd.read_csv('historical-hourly-weather-data/weather_description.csv',
                               parse_dates=[0],usecols = [0])]
    colNames = ['datetime']
    nonNumericalCols = ['datetime']
    
    # Incorportate weather conditions for each city:
    for filename in weatherFiles:
        for col, city in zip(cityIdxs, cities): 
            df = pd.read_csv(filename,usecols = [col]) 
            # Change column name from city to weather_city:
            colName = filename.replace('/','.')
            if deriv & (filename != 'historical-hourly-weather-data/weather_description.csv'): 
                colName = colName.split('.')[1] + '_deriv_' + city
            else:
                colName = colName.split('.')[1] + '_' + city
            df.columns = [colName]
            weatherData.append(df)

            # Keep an account of colunmn names and nonNumerical columns: 
            if filename == 'historical-hourly-weather-data/weather_description.csv':
                nonNumericalCols.append(colName)
            colNames.append(colName)

    weatherData = pd.concat(weatherData, axis=1, ignore_index=True)
    weatherData.columns = colNames
    
    # Exclude the first day because it's not a complete day (10-01-2012):
    weatherData = weatherData.drop(range(12))
    # Remove completely empty observations at the end of the dataset
    weatherData = weatherData.drop(range(44460,45253)) 
    weatherData.index = range(len(weatherData)) 
    # Missing data is fairly sparse, so impute missing data with last collected values
    # We could do an alternative impuation method, but using the last data point is a 
    # good method since missing data values is sporadicaly missing and because weather 
    # attributes are generally correlated from hour to hour. 
    
    # for colName in weatherData.columns:
    weatherData[colNames] = weatherData[colNames].fillna(method = 'ffill') 
        
    # z-score numerical predictors: 
    df = weatherData.drop(columns = nonNumericalCols)
    df_zscore = (df - df.mean())/df.std()
    for colName in colNames[1:-len(cities)]:
        if deriv: 
            # Derivatives can get noisy, so let's filter it:
            b, a = signal.butter(2, 0.2)
            # Time shift by the order of the filter (the first argument in butter)
            weatherData[colName] = np.append(signal.lfilter(b,a,np.diff(df_zscore[colName])) , 0)
        else:
            weatherData[colName] = df_zscore[colName]

    # Convert categorical weather types into binary representations (i.e. rainy or not)
    rainTypes = ['drizzle', 'moderate rain','light intensity drizzle', 
               'light rain', 'heavy intensity drizzle', 'heavy intensity rain',
               'light rain and snow', 'freezing rain', 'thunderstorm with rain',
               'very heavy rain', 'thunderstorm with heavy rain',
               'thunderstorm with light rain', 'squalls',
               'proximity thunderstorm with rain',
               'thunderstorm with light drizzle', 'shower rain',
               'proximity thunderstorm with drizzle',
               'light intensity shower rain','snow', 'light snow', 
               'freezing rain', 'proximity thunderstorm', 
                'thunderstorm','heavy thunderstorm']
    
    colNames = colNames[:-len(cities)]
    
    # convert weather_description into a binary column of rain or not:
    if extractRain == True:
        for city in cities:
            weatherData['rain_'+city] = weatherData['weather_description_' + city].isin(rainTypes)*1    
            weatherData = weatherData.drop(columns = ['weather_description_' + city])
            colNames.append('rain_'+city)
    display(weatherData)    
    return weatherData

def plotXCorrsBetweenTarget(weatherData, targetName = 'rain_NewYork', nCols = 3):
    ''' This function takes the weather data and plots the cross correlation between 
    the factors and the target variable, which is 'rain_NewYork' by default.

    e.g.: 

    plotXCorrsBetweenTarget(weatherData, targetName = 'rain_NewYork', nCols = 3)
    '''
    attributes = weatherData.columns[1::]
    targetVariable = weatherData[targetName]
    
    # determine cities in weatherData: 
    cities = []; 
    for i in range(len(attributes)): cities.append(attributes[i].rsplit('_',1)[1])
    cities = cities[0:int(len(cities)/6)]# 6 is the number of weather attribute types

    nAttributes = (len(attributes))//len(cities)
    nRows = ceil(nAttributes/nCols)
    
    plt.subplots(nrows = nRows, ncols = nCols, sharex='col', figsize=(15, 2*nRows))
    timeLag = range(-len(weatherData)+1,len(weatherData))
    cityColors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, len(cities)))
    gridNum=100*nRows + 10* nCols 
    
    for a in range(0,nAttributes): # iterate through each attribute
        ax = plt.subplot(gridNum+a+1)
        for c, city in zip(range(len(cities)), cities): # then by city
            attribute = attributes[len(cities)*a+c]
            # calculate Xcorr between factor and the targetVariable
            xcorr=signal.correlate(weatherData[attribute],targetVariable) / len(targetVariable)       
            ax.plot(timeLag,xcorr,linestyle = '-',marker = 'None', color = cityColors[c,:], label = city)

    # Plot formatting: 
        title = attribute.rsplit('_',1)[0]
        ax.set_title(title + ' xcorr w/' + targetName)
        ax.set_xlim((-48,48))
        ax.axvline(x=0, color = 'gray', linestyle = '--', linewidth = .5)
        ax.axhline(y=0, color = 'gray', linestyle = '--', linewidth = .5)
        if a ==0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='best')
        elif ceil((a+1)/nCols)== nRows: 
            ax.set_xlabel('Time lag (hours)')       
    plt.subplots_adjust(bottom = -.7)
    plt.show()

def constructMLdatasets(weatherData, incl_month= False):
    ''' Creates feature matrix and target vector from temporal weather data.
    The target vector is a binary object indicating if it rained the next 
    calendar day in NYC (=1) or not (=0). 
    
    In the feature matrix, each day is an observation (row) and the features 
    are weather_attribute*hour.

    e.g.: 

    featureMatrix, target, featureNames = constructMLdatasets(weatherData)
    '''
    
    attributes = weatherData.columns
    featureNames = []
    target = []
    # create hourly attributes for each measurement type: 
    for attribute in attributes[1::]:
        for hour in range(24):
            featureNames.append(attribute + '_' + str(hour))   
    if incl_month == True:
        featureNames.append('month')
        
    featureMatrix = pd.DataFrame(columns=featureNames)
    
    observationVals = []
    for idx in range(0,weatherData.shape[0]-24,24):
        target.append(any(weatherData.rain_NewYork[idx+24:idx+48])*1)
        if idx > 0:
            dict1 = dict(zip(featureNames,observationVals))
            featureMatrix = featureMatrix.append(dict1, ignore_index=True)
        observationVals = []
        for attribute in attributes[1::]:
            for hour in range(24):
                observationVals.append(weatherData[attribute][idx+hour])
        if incl_month == True:
            observationVals.append(weatherData.datetime[idx].month)
            
    dict1 = dict(zip(featureNames,observationVals))
    featureMatrix = featureMatrix.append(dict1, ignore_index=True)
    print('Feature Matrix:')
    display(featureMatrix)
    # Convert to numpyarrays for sklearn:
    featureMatrix = featureMatrix.to_numpy()
    target = target
    return featureMatrix, target, featureNames

def preprocessData(featureMatrix, target, test_size=.2, val_size =.2, n_splits=5):
    # Shuffle datasets:
    X,y = utils.shuffle(featureMatrix,target, random_state = 0) 

    # Split X and y into training and test sets (80% Train : 20% Test):
    X_Train, X_Test, y_Train, y_Test = model_selection.train_test_split(
        X, y, random_state = 0, test_size = test_size)
    # ***** N.B. Capital T (e.g. X_Train and X_Test) indicate the final training and test data.

    # Split Train sets into validation sets and training sets 
    X_train, X_val, y_train, y_val = model_selection.train_test_split(
        X_Train, y_Train, random_state = 0, test_size = val_size)

    cv=model_selection.KFold(n_splits = n_splits, shuffle = False)
    return X_Train, X_Test, y_Train, y_Test, X_train, X_val, y_train, y_val, cv

def showDataSplits(y_train, y_val,y_Train, y_Test, cv):
    ''' Helper function to show how the data was split
    
    e.g.: 
    showDataSplits(y_train, y_val,y_Train, y_Test, cv)

    '''
    fig, ax = plt.subplots(figsize = (12,3))
    plt.xlim(0,len(y_Train)+len(y_Test))
    plt.ylim(0,cv.n_splits+2.5)
    ax.set_title('Training, validation, and test splits \n (after shuffling)')
    plt.xlabel('Dataset indicies')
    yticklabels= []; 
    offset = -.4
    i = 0
    for train_idxs, cval_idxs in cv.split(y_train):
        # training data: 
        i += 1
        start = (min(train_idxs),i+offset)
        width = max(train_idxs)-min(train_idxs)
        if i == 1:
            ax.add_patch(mpl.patches.Rectangle(start, width = width, height = .8, color = 'c', label = 'CV_train'))
        ax.add_patch(mpl.patches.Rectangle(start, width = width, height = .8, color = 'c'))
        # cross-validation data: 
        start = (min(cval_idxs),i+offset)
        width = max(cval_idxs)-min(cval_idxs)
        if i == 1:
            ax.add_patch(mpl.patches.Rectangle(start, width = width, height = .8, color = 'orange', label = 'CV_validation')) 
        ax.add_patch(mpl.patches.Rectangle(start, width = width, height = .8, color = 'orange'))
        yticklabels.append('Cross validation_' + str(i))
    
    # Validation set:
    start = (0,cv.n_splits+1+offset)
    width = len(y_Train)
    ax.add_patch(mpl.patches.Rectangle(start, width = width, height = .8, color = 'yellowgreen', label = 'Validation Train')) 
    start = (len(y_train),cv.n_splits+1+offset)
    width = len(y_val)
    ax.add_patch(mpl.patches.Rectangle(start, width = width, height = .8, color = 'salmon', label = 'Validation')) 
    yticklabels.append('Validation')
    
    # Final Training and Test sets: 
    start = (0,cv.n_splits+2+offset)
    width = len(y_Train)
    ax.add_patch(mpl.patches.Rectangle(start, width = width, height = .8, color = 'g', label = 'Train')) 
    start = (len(y_Train),cv.n_splits+2+offset)
    width = len(y_Train)
    ax.add_patch(mpl.patches.Rectangle(start, width = width, height = .8, color = 'r', label = 'Test')) 
    yticklabels.append('Final test')
    
    #Format plot
    plt.yticks(np.arange(1,cv.n_splits+3),yticklabels)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.show()

def plotROC(clf,X_train,y_train, cv):
    y_train_array = np.array(y_train)
    plt.figure()
    i = 0
    AUC = []
    for train_idxs, cval_idxs in cv.split(X_train):
        mdl = clf.fit(X_train[train_idxs],y_train_array[train_idxs])
        y_score = mdl.predict_proba(X_train[cval_idxs])[:,1]
        fpr, tpr, threshold = metrics.roc_curve(y_train_array[cval_idxs], y_score, drop_intermediate = False)
        AUC.append(metrics.auc(fpr, tpr))
        plt.plot(fpr, tpr, linewidth = .75, label='ROC curve (area = %0.2f)' % AUC[i])
        i += 1
    # Annotate plot
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve, mean AUC = %.4f' % np.mean(AUC))
    plt.legend(loc="lower right")
    plt.show()

def evalROC_AUC(clf, method, X_train, y_train, AUC_scores = {}):
    if method in list(AUC_scores.keys()): 
        print('Already assessed performance of ' + method)
        return AUC_scores
    AUC_scores[method] = model_selection.cross_val_score(clf, X_train, y_train, cv=cv, scoring = 'roc_auc');
    return AUC_scores

def plotCutoff(clf,X_train,y_train, cv):
    nThresholds = 101
    thresholds = np.linspace(0,1,nThresholds)
    y_train_array = np.array(y_train)

    fig, ax = plt.subplots(figsize = (12,4))
    i = 0
    allAccuracy = np.array([])
    for train_idxs, cval_idxs in cv.split(X_train):    
        mdl = clf.fit(X_train[train_idxs],y_train_array[train_idxs])
        y_score = mdl.predict_proba(X_train[cval_idxs])[:,1]
        cvAccuracy = np.array([])
        for threshold in thresholds:
            accuracy = sum((y_score > threshold) == y_train_array[cval_idxs])/len(y_train_array[cval_idxs])
            cvAccuracy = np.append(cvAccuracy, accuracy)
        plt.plot(thresholds, cvAccuracy, linestyle = '--', linewidth = .3)
        allAccuracy = np.append(allAccuracy,cvAccuracy)
    meanAccuracy = np.mean(allAccuracy.reshape(cv.n_splits,nThresholds),axis = 0)
    plt.plot(thresholds,meanAccuracy,linewidth = 3)

    ## Annotate plot:
    yOffset = +.03
    xOffset = 0
    maxAccuracy = meanAccuracy[meanAccuracy.argmax()]
    est_Threshold = thresholds[meanAccuracy.argmax()]   

    plt.plot([est_Threshold, est_Threshold],[0, maxAccuracy], linestyle = '--', color = 'r')    
    plt.annotate('est_Threshold = %.2f'% est_Threshold, (est_Threshold-2*xOffset, .05), ha = 'right', color = 'r')
    plt.annotate('%.4f'% maxAccuracy, (est_Threshold-xOffset, maxAccuracy+yOffset), color = 'r')

    accuracy_at_50p = meanAccuracy[int(nThresholds/2)];
    plt.plot([.5, .5],[0, accuracy_at_50p], linestyle = '--', color = 'k')    
    plt.annotate('%.4f'% accuracy_at_50p, (.5+xOffset, accuracy_at_50p+3*yOffset), ha = 'center')
    plt.annotate('theoretical_Threshold = .5', (.5+xOffset, .05+2*yOffset))

    plt.xlim(0,1)
    plt.ylim(0,1)
    ax.set_title('Cutoff plot')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.show()
    retur

def plotTraining(mdlHistory, nodes = 999):
    plt.subplots(figsize=(12, 6))
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)

    key = 'val_loss'
    y = mdlHistory.history[key]
    x = list(range(len(y)))
    ax1.plot(x,y,label = key + '_' + str(nodes))

    key = 'val_accuracy'
    y = mdlHistory.history[key]
    x = list(range(len(y)))
    ax2.plot(x,y,label = key + '_' + str(nodes))
    val_acc = np.mean(mdlHistory.history['val_accuracy'][-11:-1])
    plt.annotate('%.4f'% val_acc, (x[-6],val_acc),va = 'bottom', size = 18)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='best')    
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, loc='best') 
    
    plt.show()    