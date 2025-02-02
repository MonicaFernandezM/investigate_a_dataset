#!/usr/bin/env python
# coding: utf-8

# # Project: Investigate a Dataset - [Medical-Appoinment-No-Show]
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# 
# This dataset contains information on over 100,000 medical appointments in Brazil. Each row represents a scheduled appointment and includes details about the patient, such as demographic information, medical conditions, and whether they attended their appointment or not. Understanding the factors that influence patient no-shows is crucial for improving healthcare efficiency and resource management.
# 
# #### Dataset Columns and Their Significance
# 
# | **Column Name**        | **Description** |
# |------------------------|---------------|
# | **PatientID** | Unique identifier for each patient. |
# | **AppointmentID** | Unique identifier for each appointment. |
# | **Gender** | The patient's gender (`F` = Female, `M` = Male). |
# | **ScheduledDay** | The date and time when the appointment was scheduled. |
# | **AppointmentDay** | The actual date of the appointment. |
# | **Age** | The patient's age.  |
# | **Neighbourhood** | The neighborhood where the patient resides. |
# | **Scholarship** | Indicates whether the patient is enrolled in the Brazilian social welfare program *Bolsa Família* (`1 = Yes, 0 = No`). |
# | **Hypertension** | Indicates whether the patient has been diagnosed with hypertension (`1 = Yes, 0 = No`). |
# | **Diabetes** | Indicates whether the patient has diabetes (`1 = Yes, 0 = No`). |
# | **Alcoholism** | Indicates whether the patient has a history of alcoholism (`1 = Yes, 0 = No`).  |
# | **Handcap** | Represents the presence and severity of a disability (`1, 2, 3, 4` denote different levels of impairment). |
# | **SMS_received** | Indicates whether the patient received an SMS reminder (`1 = Yes, 0 = No`). |
# | **No-show** | **Target variable.** Indicates whether the patient missed the appointment (`"No"` = Attended, `"Yes"` = Did not attend).  |
# 
# ### Question(s) for Analysis 
# 
# 1. Does sending SMS really improve attendance?
# 2. Do patients with certain medical conditions have a lower likelihood of missing appointments?
# 3. Does age or Neighborhood of residence affect attendance?
# 
# 

# In[2]:


#import packages
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# 
# ### General Properties
# 

# In[8]:


#Load Dataset
df = pd.read_csv('Database_No_show_appointments/noshowappointments-kagglev2-may-2016.csv')


# In[9]:


#returns info about the dataframe including the number of non-null values
df.info() 


# With df.info(), we can see the number of non-null values in each column and the data type of each one.

# In[10]:


#returns the shape of the DataFrame
df.shape


# In[11]:


#returns number of unique values for each column 
df.nunique()


# By looking at the unique values, we can see that there are 62,299 unique patientId values and 110,527 unique appointmentID values, which is the total number of rows I have. Here, we can observe that the same patient has had more than one appointment scheduled.

# In[7]:


#returns summary statistics
df.describe()


# **1.Age:**
# The age range goes from -1 (which could indicate a missing or incorrect value) up to 115 years.The mean age is 37 years, with a standard deviation of 23 years, suggesting a wide diversity in patients’ ages. There are lower values (25%) that likely correspond to younger patients.
# 
# **2.Scholarship:**
# The mean is low (0.098), indicating that only a small percentage of patients are part of the Brazilian social assistance program (Bolsa Família).The column contains binary values (0 or 1), and 75% of patients are not part of this program.
# 
# **3.Hypertension:**
# Approximately 20% of patients have hypertension, as the mean is 0.197.
# 
# **4.Diabetes:**
# Only 7.18% of patients have diabetes, with a mean of 0.072.
# 
# **5.Alcoholism:**
# Only 3.04% of patients have a history of alcoholism, suggesting that this condition is less prevalent in the dataset.
# 
# **6.Handcap (Disability):**
# The values in this column are integers between 0 and 4, which could indicate levels of disability.
# 
# **7.SMS_received:**
# The mean is 0.32, indicating that about 32% of patients received an SMS reminder.

# 
# ### Data Cleaning
# > **Tip**: Make sure that you keep your reader informed on the steps that you are taking in your investigation. Follow every code cell, or every set of related code cells, with a markdown cell to describe to the reader what was found in the preceding cell(s). Try to make it so that the reader can then understand what they will be seeing in the following cell(s).
#  

# In[12]:


#We check if we have a duplicated rows
sum(df.duplicated())


# We see that we don’t have any duplicate rows.

# In[40]:


#We check if we have patientID duplicated 
sum(df.PatientId.duplicated())


# But we do have the same PatientID repeated, as the same patient could have multiple appointments.

# In[13]:


#Count the number of appointments by 'PatientId'
appointment_counts = df['PatientId'].value_counts()

#Filter the 'PatiendId' with more than one scheduled appointment
multiple_appointments = appointment_counts[appointment_counts > 1]
multiple_appointments


# We can see which PatientId has more than one scheduled appointment.

# In[17]:


#Filter the rows where 'Age' has a negative value
negative_age = df[df['Age'] < 0]
negative_age


# We only have one row that contains this value. We can check if this PatientId is repeated to see if we can replace the age with the correct data or remove the row.

# In[19]:


#Filter the rows where 'Age' is -1
incorrect_age = df[df['Age'] == -1]

#Get the 'PatientId' of those rows
patient_id = incorrect_age['PatientId'].iloc[0] 

#Check if this 'PatientId' is repeated and get the associated ages
repeated_patient = df[df['PatientId'] == patient_id]

#View the ages associated with that PatientId
print(repeated_patient[['PatientId', 'Age']])


# Since we cannot determine the age of this patient and it is only one data point out of 110527, we will remove this row.

# In[24]:


#remove row with 'Age'== -1
df = df[df['PatientId'] != patient_id]


# In[25]:


#Correct data tipe 'ScheduledDay' and 'AppointmentDay'
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])


# We will change the data type of ScheduledDay and AppointmentDay, as they are dates and should be of the datetime type.

# In[23]:


df.info()


# Now, with df.info(), we can see that the number of rows is 110526 and that the data for ScheduledDay and AppointmentDay are of datetime type.

# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# > **Tip**: Now that you've trimmed and cleaned your data, you're ready to move on to exploration. **Compute statistics** and **create visualizations** with the goal of addressing the research questions that you posed in the Introduction section. You should compute the relevant statistics throughout the analysis when an inference is made about the data. Note that at least two or more kinds of plots should be created as part of the exploration, and you must  compare and show trends in the varied visualizations. Remember to utilize the visualizations that the pandas library already has available.
# 
# 
# 
# > **Tip**: Investigate the stated question(s) from multiple angles. It is recommended that you be systematic with your approach. Look at one variable at a time, and then follow it up by looking at relationships between variables. You should explore at least three variables in relation to the primary question. This can be an exploratory relationship between three variables of interest, or looking at how two independent variables relate to a single dependent variable of interest. Lastly, you  should perform both single-variable (1d) and multiple-variable (2d) explorations.
# 
# 
# ### Does sending SMS really improve attendance?

# In[ ]:


# Use this, and more code cells, to explore your data. Don't forget to add
#   Markdown cells to document your observations and findings.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Research Question 2  (Replace this header name!)

# In[ ]:


# Continue to explore the data to address your additional research
#   questions. Add more headers as needed if you have more questions to
#   investigate.


# <a id='conclusions'></a>
# ## Conclusions
# 
# > **Tip**: Finally, summarize your findings and the results that have been performed in relation to the question(s) provided at the beginning of the analysis. Summarize the results accurately, and point out where additional research can be done or where additional information could be useful.
# 
# > **Tip**: Make sure that you are clear with regards to the limitations of your exploration. You should have at least 1 limitation explained clearly. 
# 
# > **Tip**: If you haven't done any statistical tests, do not imply any statistical conclusions. And make sure you avoid implying causation from correlation!
# 
# > **Tip**: Once you are satisfied with your work here, check over your report to make sure that it is satisfies all the areas of the rubric (found on the project submission page at the end of the lesson). You should also probably remove all of the "Tips" like this one so that the presentation is as polished as possible.
# 
# ## Submitting your Project 
# 
# > **Tip**: Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should see output that starts with `NbConvertApp] Converting notebook`, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > **Tip**: Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > **Tip**: Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[ ]:


# Running this cell will execute a bash command to convert this notebook to an .html file
get_ipython().system('python -m nbconvert --to html Investigate_a_Dataset.ipynb')

