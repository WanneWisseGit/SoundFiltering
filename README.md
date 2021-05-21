**SoundFiltering**

During my graduation project I needed to cancel noise of a ventilator.

This noise is constant so no need to do time-critical modeling. 
A constant noise is a noise which frequencies repeat themselfs in a timespan.

online_modeling.py and offline_modeling.py can model the anti-noise for a constant noise in a non time-critical way. 

In offline_modeling.py a method is implemented to cancel a constant noise offline.

In online_modeling.py a method is implemented to cancel a constant noise online.

Hardware requirements
- PC can be very slow, adjust output queue size accordingly.
- Speaker to play anti-noise: should be able to play the frequencies of the noise.
- Microphone to record: shoud be able to pickup the frequencies of the noise.

Result Online modeling graduation assignment

![image](https://user-images.githubusercontent.com/36732907/119158621-b285d080-ba56-11eb-9a36-d28ba7fd3927.png)

We can see that after 10 seconds the noise is cancelled at the point of the microphone and stays inphase with the noise.

The top graph shows the frequencies and their energy(fft of the main frequencies) while the bottom graph shows the total error calculated by sum(square(one_second_of_audio))
