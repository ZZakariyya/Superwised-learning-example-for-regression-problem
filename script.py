import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.offline as pyo

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import classification_report


st.set_page_config(layout="wide")

page = st.sidebar.selectbox(
    'Pages',
    ('Home Page', 'EDA', 'Feature Selection and Modelling')
)

if page == 'Home Page':
    col1, col2 = st.columns(2)

    with col1:
        st.header("What is supervised learning?")
        st.write("Supervised learning, also known as supervised machine learning, is a subcategory of machine learning and artificial intelligence. It is defined by its use of labeled datasets to train algorithms that to classify data or predict outcomes accurately. As input data is fed into the model, it adjusts its weights until the model has been fitted appropriately, which occurs as part of the cross validation process. Supervised learning helps organizations solve for a variety of real-world problems at scale, such as classifying spam in a separate folder from your inbox.")
        st.subheader("How supervised learning works")
        st.write("Supervised learning uses a training set to teach models to yield the desired output. This training dataset includes inputs and correct outputs, which allow the model to learn over time. The algorithm measures its accuracy through the loss function, adjusting until the error has been sufficiently minimized.")
        st.write("Supervised learning can be separated into two types of problems when data mining—classification and regression:")
        st.markdown(" ***-Classification uses an algorithm to accurately assign test data into specific categories. It recognizes specific entities within the dataset and attempts to draw some conclusions on how those entities should be labeled or defined. Common classification algorithms are linear classifiers, support vector machines (SVM), decision trees, k-nearest neighbor, and random forest, which are described in more detail below.*** ")
        st.markdown(" ***-Regression is used to understand the relationship between dependent and independent variables. It is commonly used to make projections, such as for sales revenue for a given business. Linear regression, logistical regression, and polynomial regression are popular regression algorithms.*** ")
        
    with col2:
        image = Image.open('Capture.png')
        image = image.resize((800,500))
        st.image(image, caption='Picture is taken from Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron')

    for i in range(5):
        st.write("##")
    col1, col2, col3 = st.columns(3)

    with col2:
        st.subheader("What is this webpage used for:")
        st.write("In this project i made 3 webpages which all of them does different steps of supervised machine learning project process")
        st.write("##")
        st.subheader("***_Home Page_***")
        st.write("Just information about Supervised Learning and explains different pages from this webpage")
        st.write("##")
        st.subheader("***_EDA_***")
        st.write("EDA is done here. You can upload your data and do simple data cleaning, some simple visualisations to get to know the data.")
        st.write("##")
        st.subheader("***_Modelling_***")
        st.write("You can do your feature selection and modelling on this page. And of course you can check your metrics and stuff")
elif page == 'EDA':
    uploaded_file = st.file_uploader("Choose a csv file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df, width = 1000, height = 500)

        for i in range(3):
            st.write("##")

        check1 = st.checkbox(
            'Click to see your data info '
        )

        if check1:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("Null values in your data")
                st.text(df.isnull().sum())
                st.write(r"Sum of all null values are : {sum}".format(sum  = df.isnull().sum().sum()))

            with col2:
                st.write("Data types in your data")
                st.text(df.dtypes)
                st.write(r"Shape of your data is {shape}".format(shape = df.shape))

            with col3:
                st.write("Description of your data")
                st.write(df.describe().T)

        select1 = st.selectbox(
            'What you wanna see in your data?',
            ("None", "See value counts of the column you choose", "See outliers with boxplots",
            "See correlation in your data with the help of scatter plots")
        )

        if select1 == "See value counts of the column you choose":
            
            tup = tuple(df.columns)

            select2 = st.selectbox(
                'Pick a column',
                tup
            )

            col1, col2 = st.columns(2)

            with col1:
                check1 = st.checkbox(
                    'Pie Chart'
                )
                if check1:
                    values = []
                    for i in list(df[select2].unique()):
                        count = 0
                        for j in range(len(df[select2])):
                            if df[select2][j] == i:
                                count +=1
                        values.append(count)

                    fig = go.Figure(data=[go.Pie(labels=df[select2].unique(), values=values)])

                    colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']

                    fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                                marker=dict(colors=colors, line=dict(color='#000000', width=2)))

                    fig.update_layout(
                        autosize=False,
                        width=800,
                        height=800,)

                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                check2 = st.checkbox(
                    'Countplot'
                )

                if check2:
                    fig = plt.figure(figsize=(10, 4))
                    sns.countplot(x = df[select2], data = df)
                    st.pyplot(fig)
        elif select1 == "See outliers with boxplots":
            numerical_variables = [i for i in df.columns if df[i].dtype == "int64" or df[i].dtype == "float64"] 
            tup = tuple(numerical_variables)
            var = st.selectbox(
                "Please pick your variable",
                tup
            )


            fig = px.box(df, y = var)

            fig.update_layout(
            autosize=False,
            width=700,
            height=700,)

            st.plotly_chart(fig, use_container_width=True)
        elif select1 == "See correlation in your data with the help of scatter plots":

            numerical_variables = [i for i in df.columns if df[i].dtype == "int64" or df[i].dtype == "float64"]
            col1, col2, col3 = st.columns(3)
            tup = tuple(numerical_variables)
            with col1:
                val1 = st.selectbox(
                    "Please pick your X axis variable",
                    tup
                )

            with col2:
                val2 = st.selectbox(
                    "Please pick your Y axis variable",
                    tup
                )

            with col3:
                hue = st.selectbox(
                    "Please pick your hue variable",
                    tup
                )

            fig = px.scatter(df, x=val1, y=val2, color = hue)

            fig.update_layout(
                autosize=False,
                width=1200,
                height=800,)

            st.plotly_chart(fig, use_container_width=True)
elif page == 'Feature Selection and Modelling':
    
    st.title("Data Preparation, Feature Selection and Modelling")
    
    uploaded_file = st.file_uploader("Choose a csv file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df, width = 1000, height = 500)

        for i in range(2):
            st.write("##")

        st.header("Data Preparation")
        st.write("##")

        st.subheader("Dealing with Null Values")
        st.write("##")

        select5 = st.selectbox(
                'What do you want to do with null values',
                ("None", "Delete them", "Fill them with mean")
            )

        if select5 == "Delete them":
            k = df.isnull().sum().sum()
            df.dropna(inplace = True, axis = 1)
            st.write(r"Sum of null values before removing null is {before} and after it is {after}".format(before = k, after = df.isnull().sum().sum()))
            
        elif select5 == "Fill them with mean":
            df.fillna(df.mean())
            st.write("Succesfully filled with the mean")
        
        

        st.subheader("Dealing with Outliers")
        st.write("##")


        select3 = st.selectbox(
                'What do you want to do with outliers',
                ("None", "Delete them", "Clip them to lowerbound and upperbound")
            )

        numerical_variables = [i for i in df.columns if df[i].dtype == "int64" or df[i].dtype == "float64"] 
        tup = tuple(numerical_variables)
        options = st.multiselect(
            "Select your column(s)",
            tup
        )
      
        if select3 ==  "Delete them":
            k = df.shape
            
            for i in options:
                upper_i = df[i].mean() + 3*df[i].std()
                lower_i = df[i].mean() -3*df[i].std()

                df = df[(df[i]<upper_i) & (df[i]>lower_i)]
            
            st.write(f"Data Shape before removing outliers is {k} and after it is {df.shape}")
        elif select3 == 'Clip them':
            for i in options:
                upper_i = df[i].mean() + 3*df[i].std()
                lower_i = df[i].mean() -3*df[i].std()

                df = df.clip(lower_i, upper_i)

        st.subheader("Encoding Categorical Variables")
        st.write("##")      

        categorical_variables = [i for i in df.columns if df[i].dtype == 'object'] 
        
        tup2 = tuple(categorical_variables)

        options2 = st.multiselect(
            "Select your column",
            tup2
        )
        check1 = st.checkbox(
            'Label Encode'
        )

        if check1:
            df[options2] = df[options2].apply(LabelEncoder().fit_transform)

        categorical_variables += " "
        tup3 = tuple(categorical_variables)

        options2 = st.multiselect(
            "Select your column",
            tup3
        )

        check2 = st.checkbox(
            'One Hot Encode'
        )

        if check2:
            try :
                enc=OneHotEncoder()
                enc_data=pd.DataFrame(enc.fit_transform(df[options2]).toarray())
                df=df.join(enc_data)
                df.drop(options, inplace = True, axis = 1)
            except : 
                st.write("Oops, wrong columns name!!")
                st.write("Psst, reason behind there is space among column names is because streamlit doesnt support 2 exact same mutiselectboxes.")

        st.dataframe(df, width = 1000, height = 500) 
        
        st.write("##")
        st.subheader("Feature Selection")
        st.write("##") 
            
        tup4 = tuple(df.columns)

        col1, col2 = st.columns(2)

        with col1:
            options3 = st.multiselect(
                "Select your features",
                tup4
            )

        with col2:
            select5 = st.selectbox(
                'Select your target',
                tup4
            )

        X = df[options3]
        y = df[select5]

        check7 = st.checkbox(
            'Split data to train and test'
        )

        if check7:
            size = st.number_input('Insert a test size')
            st.write("Remember test size should be in (0,1) interval!!!")

            state = st.number_input('Insert a random state number')
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=size, random_state=int(state))

            st.write(f"Your train size for features is {X_train.shape} and target {y_train.shape}")
            st.write(f"Your test size for features is {X_test.shape} and target {y_test.shape}")


        st.write("##")
        st.subheader("Scaling the data")
        st.write("##") 

        select8 = st.selectbox(
            'Which scaling method you want for your training data',
            ("None", "Standart Scale", "Robust Scale")
        )

        if select8 == "Standart Scale":
            scaler = StandardScaler()
            X_col = X_train.columns
            X_train = scaler.fit_transform(X_train)
            X_train = pd.DataFrame(X_train, columns = X_col)

        if select8 == "Robust Scale":
            scaler = RobustScaler()
            X_col = X_train.columns
            X_train = scaler.fit_transform(X_train)
            X_train = pd.DataFrame(X_train, columns = X_col)

        st.dataframe(X_train)

        st.write("##")
        st.header("Modelling")
        st.write("Finally")
        st.write("##") 

        select9 = st.selectbox(
            'Which type of supervised machine learning task are your working on?',
            ("", "Classification", "Regression")
        )

        if select9 == "Regression":
            st.write("We are gonna implement linear regression for this task")

            lr = LinearRegression().fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            st.write(f"R2 score is {r2_score(y_test, y_pred):.2f}")
            st.write(f"Mean Squared Error is {mean_squared_error(y_test, y_pred):.2f}")
            st.write(f"Mean Absolute Error is {mean_absolute_error(y_test, y_pred):.2f}")
        elif select9 == "Classification":
            select10 = st.selectbox(
                'Which Classification model do you want?',
                ("", "Logistic Regression", "Random Forest", "Xgboost")
            )

            if select10 == "Logistic Regression":
                model = LogisticRegression().fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.write(f"Your Acurracy Score is {accuracy_score(y_test, y_pred):.2f}")
                st.write("Classification Report")
                st.text(classification_report(y_test, y_pred))
                st.write("Confusion Matrix")
                fig = ff.create_annotated_heatmap(confusion_matrix(y_test, y_pred), x = ["Postive", "Negative"], y = ["Negative", "Postive"])

                st.plotly_chart(fig)
            elif select10 == "Random Forest":
                model = RandomForestClassifier().fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                st.write(f"Your Acurracy Score is {accuracy_score(y_test, y_pred):.2f}")
                st.write("Classification Report")
                st.text(classification_report(y_test, y_pred))
                st.write("Confusion Matrix")
                fig = ff.create_annotated_heatmap(confusion_matrix(y_test, y_pred), x = ["Postive", "Negative"], y = ["Negative", "Postive"])

                st.plotly_chart(fig)
            elif select10 == "Xgboost":
                model = XGBClassifier().fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                st.write(f"Your Acurracy Score is {accuracy_score(y_test, y_pred):.2f}")
                st.write("Classification Report")
                st.text(classification_report(y_test, y_pred))
                st.write("Confusion Matrix")
                fig = ff.create_annotated_heatmap(confusion_matrix(y_test, y_pred), x = ["Postive", "Negative"], y = ["Negative", "Postive"])

                st.plotly_chart(fig)