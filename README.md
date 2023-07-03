# Parkinson_Prediction

Comparing methods for predcting Parkinson's Disease and determining the most significant variables for prediction

Parkinson’s disease is a brain disorder that causes unintended or uncontrollable movements, such as shaking, stiffness, and difficulty with balance and coordination.

Symptoms usually begin gradually and worsen over time. As the disease progresses, people may have difficulty walking and talking. They may also have mental and behavioral changes, sleep problems, depression, memory difficulties, and fatigue.

Older woman and her caregiverWhile virtually anyone could be at risk for developing Parkinson’s, some research studies suggest this disease affects more men than women. It’s unclear why, but studies are underway to understand factors that may increase a person’s risk. One clear risk is age: Although most people with Parkinson’s first develop the disease after age 60, about 5% to 10% experience onset before the age of 50. Early-onset forms of Parkinson’s are often, but not always, inherited, and some forms have been linked to specific alterations in genes.

Speech difficulties (dysarthria) and voice problems are very common in people with Parkinson’s disease. Of the more than seven million people with Parkinson’s disease worldwide, between 75% and 90% will develop voice and speech problems over the course of their illness.

If one is affected by Parkinson’s disease, some of the voice and speech difficulties seen include:

Softened voice. Reduced volume to the voice.
Speaking in an unchanging pitch (monotone).
Having a hoarse or strained voice.
Having a breathiness to the voice. Breathiness in the quality of voice that is easily heard by your listeners. It takes more effort and energy to speak. One breathes heavily while speaking
Trouble clearly and easily pronouncing letters and words.
Tremor in the voice.
Slurring of speech.
Using short rushes of speech.
Loss of facial expression.

Training data source : UCI ML Parkinson's disease dataset
Attributes/variables in the dataset:
name - ASCII subject name and recording number

MDVP:Fo(Hz) - Average vocal fundamental frequency

MDVP:Fhi(Hz) - Maximum vocal fundamental frequency

MDVP:Flo(Hz) - Minimum vocal fundamental frequency

MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP - Several measures of variation in fundamental frequency

MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA - Several measures of variation in amplitude

NHR,HNR - Two measures of ratio of noise to tonal components in the voice

status - Health status of the subject (one) - Parkinson's, (zero) - healthy

RPDE,D2 - Two nonlinear dynamical complexity measures

DFA - Signal fractal scaling exponent

spread1,spread2,PPE - Three nonlinear measures of fundamental frequency variation

Models used: Logistic Regression, Support Vector Machines, Random Forest Classifier, Neural Network Applications
