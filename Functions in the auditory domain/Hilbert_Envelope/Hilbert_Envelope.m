function [hilbert_envelope_value] = Hilbert_Envelope(x)

    % This function is an implementation of the Hilbert-Envelope sound
    % intensity function
    
    % -------
    % Authors: Hilbert, David (1912) 'Grundzüge einer allgemeinen Theorie der linearen Integralgleichungen' 
    % doi: https://archive.org/details/grundzgeeinera00hilbuoft

    % Input stream: The momentary air pressure pressure (Pa) at the
    % tympanic membrane ('x').
    
    % -----
    % Main Parameters
    % N/A
    

    % Calculate momentary discrete-time analytic signal using Matlab's in-built 
    % Hilbert function (http://uk.mathworks.com/help/signal/ref/hilbert.html)
    
    analytic_signal = hilbert(x);
    
    
    % Take the absolute value of analytic signal to generate
    % envelope
    
    hilbert_envelope_value = abs(analytic_signal);
    
end



