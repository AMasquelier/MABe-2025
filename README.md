# Context
When I first started working on this competition, my goal was simply to get my hands on a type of problem that I never had the opportunity to work on: segmentation. My objective was to build a CNN model to at least reach the top 50% and beat those XGBoost public notebooks that got too popular to my taste.<br>
Luckily I ended up at the 103rd place out of 1413 in the leaderboard.<br>
The things that I've learned by working on this competition were:
- How to build segmentation models with CNNs
- Simplicity is key


# Data preprocessing
The data in this competition was ... challenging to work with. There were different labs with different tracking systems and different target behaviors. The most important part was then to find a way to transform this overwelming diversity of data into something simple.<br>
There were around 25 bodyparts that were tracked by the different labs but a lot of them were used in a few recordings, so I only kept the most used ones and merged the bodyparts that were almost the same (e.g. hips and lateral; I also noticed later that I forgot to merge head and nose/neck which means that I had headless mice in 17 recordings *facepalm*).<br>
Then I created .npy files containing the coordinates of the 7 bodyparts I kept (nose, neck, lateral_left/right, ear_left/right and tail_base) for each pair of mice and individual mouse with the corresponding labels.


# One model to predict them all
For the sake of simplicity, the model should predict all the different behaviors at the same time and handle any lab and solo/social behaviors. <br>
The features used are: 
- Speed norm of mouse A and B
- The cross product of the between the acceleration and the speed of mouse A and B
- The cosine similarity between each frame of the speed of mouse A and B (capture the changes of direction between frames)
- The cosine similarity between the speed of mouse A and B
- The distance between each pair of bodyparts of mouse A and each pair of bodyparts of mouse B
- The derivative of the distance
- The cosine similarity between the speed and the difference of each mice's bodypart positions for mouse A and B<br>

NB: In the case of solo recording mouse A is the same as mouse B

# Loss function
The loss function is simply a Binary Cross Entropy function for each frame. The "behaviors labeled" column from "train.csv" is used to nullify the loss on behaviors that are not tracked to not "punish" the model for predicting a behviour that may be correct but that was simply not annotated.
