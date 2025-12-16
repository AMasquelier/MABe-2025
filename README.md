# Context
When I first started working on this competition, my goal was simply to get my hands on a type of problem that I never had the opportunity to work on: segmentation. My objective was to build a CNN model to at least reach the top 50% and beat those XGBoost public notebooks that got too popular to my taste.
Luckily I ended up at the 103rd place out of 1413 in the leaderboard (top 7%).
The things that I learned by working on this competition were:
- How to build segmentation models with CNNs
- Simplicity is key


# Data preprocessing
The data in this competition was ... challenging to work with. There were different labs with different tracking systems and different target behaviors. The most important part was then to find a way to transform this overwelming diversity of data into something simple.
There were around 25 bodyparts that were tracked by the different labs but a lot of them were used in a few recordings, so I only kept the most used ones and merged the bodyparts that were almost the same (e.g. hips and lateral; I also noticed later that I forgot to merge head and nose/neck which means that I had headless mice in 17 recordings *facepalm*).
Then I created .npy files containing the coordinates of the 7 bodyparts I kept (nose, neck, lateral_left/right, ear_left/right and tail_base) for each pair of mice and individual mouse with the corresponding labels.


# One model to predict them all
