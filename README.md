"SoundFiltering

During my graduation project I needed to cancel noise of a ventilator.

This noise is constant so no need to do time-critical modeling. 
A constant noise is a noise which frequencies repeat themselfs in a timespan.

online_modeling.py and offline_modeling.py can model the anti-noise for a constant noise in a non time-critical way. 

In offline_modeling.py a method is implemented to cancel a constant noise offline.

In online_modeling.py a method is implemented to cancel a constant noise online.
Hardware requirements:
    PC can be very slow, adjust output queue size accordingly
    Speaker to play anti-noise: should be able to play the frequencies of the noise
    Microphone to record: shoud be able to pickup the frequencies of the noise

" 
