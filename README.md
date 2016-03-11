# third-party-functions
This repository holds the source code of functions written by third-party research groups, for use as hypotheses in the Kymata Atlas. *Note: This repository only holds functions for which there is not already an existing Github repository.*

Where possible, the code should follow - exactly - a published description. This description, with a doi (or a pdf of the original description itself), should be cited, and ideally the code should contain comments that link lines of code to equation numbers eg.

```javascript
// This function is taken from J. Doe (2002), 'Hypothesized  sound to pitch model' J. Neuroscience
// doi://12345.678910
// Input stream is binaural audition, 20ms history window.

Doe02_sound_to_pitch(left_channel,right_channel){

  // Implements eq. 1

  x = compress((left_channel)^4,(right_channel)^4);
  
  // Implements eq. 2

  for(i = 0; i < 5; i++){ 
    y = x[i];
    
    etc ...

```
