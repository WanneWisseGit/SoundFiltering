#TODO create a errorplot class
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import time
import queue
import keyboard
import enum
from scipy.io.wavfile import write
import os

#class used to set the direction of shift or amplitude
class Direction(enum.Enum):
   Right = 1.0
   Left = -1.0

#class to cancel the frequencies
class AntiSoundController:
    def __init__(self,sample_size,sampling_rate,amount_frequencies_to_cancel):
        #list of frequencies to cancel
        self.sampling_rate = sampling_rate
        self.sample_size = sample_size
        self.amount_frequencies_to_cancel = amount_frequencies_to_cancel
        #filled after init
        self.frequencies_to_cancel = None
        #variable to declare init
        self.init_done = False
        #queue with data of pyaudio
        self.input_queue = queue.Queue()
        #queue with data for pyaudio
        self.output_queue = queue.Queue()
        #time when to change the amplitude
        self.time_till_change_amplitude = 0
        #sample of one second of pyaudio
        self.microphone_samples_one_second = np.array([])
        #costcalculator to determine the energy of the frequencies which is cost
        self.costcalculator = CostCalculator(sampling_rate)
        #list of sinus generators, shift correcters and amplitude correcters one per frequency to cancel
        self.sinus_generators =  [SinusGenerator(sample_size,sampling_rate) for i in range(self.amount_frequencies_to_cancel)]
        self.amplitude_correcters = [Correcter(0.005,sampling_rate,self.costcalculator) for i in range(self.amount_frequencies_to_cancel)]
        #pyaudio object to deal with pyaudio
        self.pyaudio_object = pyaudio.PyAudio()
        #stream from pyaudio to interact with speakers and microphone
        self.stream = self.pyaudio_object.open(channels=1,
                    frames_per_buffer= sample_size,
                    rate=sampling_rate, 
                    format=pyaudio.paFloat32, 
                    output=True,
                    input=True,
                    output_device_index=3,
                    input_device_index=1,
                    stream_callback=self.callback)
    
    #callback function used by pyaudio to get new data
    def callback(self, in_data, frame_count, time_info, status): 
        
        #data to input queue
        self.input_queue.put(in_data)
        #data from output queue to pyaudio
        output = self.output_queue.get()
        return (output, pyaudio.paContinue)
    
    def append_numpy_to_file(self,one_second):
        with open('test.npy', 'ab') as f:
            np.save(f, one_second)

    def write_numpy_to_wav(self):
        with open('test.npy', 'rb') as f:
            fsz = os.fstat(f.fileno()).st_size
            out = np.load(f)
            while f.tell() < fsz:
                out = np.concatenate([out,np.load(f)])
            write("recorded_18-05-2021.wav",self.sampling_rate, out.astype(np.float32))
    
    #function to start the stream
    def start_stream(self):
        self.stream.start_stream() 
    
    #init function to get the main frequencies from the sound
    def get_main_frequencies(self):
      
        if not self.input_queue.empty():
            #get the inputdata as float 32
            microphone_wave = self.input_queue.get()
            microphone_wave = np.frombuffer(microphone_wave, dtype=np.float32)
            #create 1 second of microphone_waves
            self.microphone_samples_one_second = np.concatenate([self.microphone_samples_one_second, microphone_wave])
            #when we really have one second, analyse and get main frequencies
            if len(self.microphone_samples_one_second) >= self.sampling_rate*10:
                #self.microphone_samples_one_second = self.microphone_samples_one_second[:self.sampling_rate]
                frequencies,energy = self.costcalculator.calculate_frequencies_energy(self.microphone_samples_one_second)
                top_indices = self.costcalculator.get_top_N_indices_energy(energy,AMOUNT_FREQUENCIES_TO_CANCEL)
                frequencies_to_cancel = frequencies[top_indices]
                self.shift_correcters = []
                for frequency in frequencies_to_cancel:
                    self.shift_correcters.append(Correcter(0.0001, self.sampling_rate,self.costcalculator))
                
                self.microphone_samples_one_second = np.array([])
                self.input_queue.queue.clear()
                self.output_queue.queue.clear()
                self.init_done = True
                return frequencies_to_cancel
        #play a silent sound to let the stream play and input sound
        if self.output_queue.qsize()<5:
            silent_wave = np.zeros(self.sample_size).astype(np.float32)
            self.output_queue.put(silent_wave.tobytes())
        else:
            time.sleep(0.01)
    
    #main function cancel the main frequencies by changing shift and amplitude   
    def cancel_main_frequencies(self):
        
        #cancel program and show error
        if keyboard.is_pressed('a'):
            self.plot(ERROR_LIST, TOTAL_ERROR_LIST)
            self.write_numpy_to_wav()
            return
        #we have got input data from pyaudio
        if not self.input_queue.empty():
            #get the inputdata as float 32
            microphone_wave = self.input_queue.get()
            microphone_wave = np.frombuffer(microphone_wave, dtype=np.float32)
            #create 1 second of microphone_waves
            self.microphone_samples_one_second = np.concatenate([self.microphone_samples_one_second, microphone_wave])
            #when we really have one second update shift and amplitude
            if len(self.microphone_samples_one_second) >= self.sampling_rate:
                #self.microphone_samples_one_second = self.microphone_samples_one_second[:self.sampling_rate]
                #for every frequency update shift
                self.append_numpy_to_file(self.microphone_samples_one_second)
                total_error = np.sum(np.square(self.microphone_samples_one_second))
                TOTAL_ERROR_LIST.append(total_error)
                for index, frequency in enumerate(self.frequencies_to_cancel):
                    error, shift = self.shift_correcters[index].get_corrected_value(self.microphone_samples_one_second, frequency)
                    ERROR_LIST[index].append(error)
                    self.sinus_generators[index].change_shift(shift)
                #update amplitude only once every 30 seconds 
                if self.time_till_change_amplitude <= 30:
                    self.time_till_change_amplitude += 1
                else: 
                    #for every frequency update amplitude(double for maybe swap logic)
                    for index, frequency in enumerate(self.frequencies_to_cancel):
                        error, amplitude = self.amplitude_correcters[index].get_corrected_value(self.microphone_samples_one_second,frequency)
                        self.sinus_generators[index].change_amplitude(amplitude)
                    #reset time to 10 to get the shift right
                    self.time_till_change_amplitude = 10
                #we have parsed one second of data, now clear variable and wait untill new
                self.microphone_samples_one_second = np.array([])
        #when we have less then 10 samples in out queue put in new samples with the right amplitude and shift
        if self.output_queue.qsize()<5:
            #print(self.output_queue.qsize())
            #create a summed wave 
            summed_wave = np.zeros(self.sample_size).astype(np.float32)
            #sum all frequencies with right amplitude and shift
            for index, frequency in enumerate(self.frequencies_to_cancel):
                summed_wave += self.sinus_generators[index].get_sample(frequency).astype(np.float32)
            #append to output queue so pyaudio can callback for it
            self.output_queue.put(summed_wave.tobytes())
        else:
            time.sleep(0.01)
    
    #main logic program
    def main(self):
        while self.stream.is_active():
            
            if self.init_done == False:
                if os.path.exists("test.npy"):
                    os.remove("test.npy")
                self.frequencies_to_cancel = self.get_main_frequencies()
            else:
                self.cancel_main_frequencies()
            
    #plot error of every frequency   
    def plot(self, error,total_error):
        fig, axs = plt.subplots(2)
        fig.suptitle('Error')
        for e in error:
            axs[0].plot(e)
        axs[1].plot(total_error)
        plt.show()
    
    #close stream
    def stop_stream(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio_object.terminate()

#class to calculate the cost of frequencies
class CostCalculator:
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate
    #get all frequencies with their energy
    def calculate_frequencies_energy(self, microphone_samples_one_second):
        frequencies = np.fft.rfftfreq(microphone_samples_one_second.size, d=1/self.sampling_rate)
        energy = np.fft.rfft(microphone_samples_one_second)
        energy = np.abs(energy)
        energy = np.square(energy)
        energy = energy
        return (frequencies,energy)
    
    #get top frequencies from energy
    def get_top_N_indices_energy(self, energy, N):
        indices = energy.argsort()[-N:][::-1]
        return indices[:N]
    
    #get energy for a specific frequency
    def get_energy_frequency(self, frequencies, energies, frequency):
        absolute_val_array = np.abs(frequencies - frequency)
        smallest_difference_index = absolute_val_array.argmin()
        return energies[smallest_difference_index]

#class to correct shift or amplitude of one frequency
class Correcter:
    def __init__(self,learning_rate, sampling_rate, costcalculator):
        self.learning_rate = learning_rate
        self.previous_error = 1000
        self.biggest_error = 0
        self.direction = Direction.Right
        self.sampling_rate = sampling_rate
        #use costcalculator class
        self.costcalculator = costcalculator
    #change amplitude or shift according to the cost for a frequency calculated by costcalculator
    def get_corrected_value(self, microphone_samples_one_second, frequency_to_correct):
        frequencies, energies = self.costcalculator.calculate_frequencies_energy(microphone_samples_one_second)
        current_error = self.costcalculator.get_energy_frequency(frequencies,energies,frequency_to_correct)
    
        if current_error>self.previous_error:
            if self.direction == Direction.Right:
                self.direction = Direction.Left
            else:
                self.direction = Direction.Right
        self.previous_error = current_error
        corrected_value = self.direction.value * self.learning_rate
        return current_error, corrected_value
        
#Class to create sinus of one frequency
class SinusGenerator:
    def __init__(self,sample_size,sampling_rate):
        self.time = 0.0
        self.sample_size = sample_size
        self.sampling_rate = sampling_rate
        self.amplitude = 0.1
    #get sample given a frequency and changed amplitude and shift
    def get_sample(self,frequency):
        time_sound_sample = np.linspace(self.time,self.time+((self.sample_size - 1)/self.sampling_rate),self.sample_size)
        self.time += self.sample_size/self.sampling_rate 
        return self.amplitude* np.sin(2*np.pi*frequency*time_sound_sample).astype(np.float32)

    #change shift which is done by changing time
    def change_shift(self, seconds):
        self.time += seconds
    
    #change amplitude
    def change_amplitude(self, amount_amplitude):
        self.amplitude += amount_amplitude

#samplesize for pyaudio stream
SAMPLESIZE = 1000
#how many samples per second in pyaudio stream
SAMPLINGRATE = 44000
#the amount of frequencies to cancel from noise
AMOUNT_FREQUENCIES_TO_CANCEL = 1
#list per frequency
ERROR_LIST = [[] for i in range(AMOUNT_FREQUENCIES_TO_CANCEL)]
#list total error
TOTAL_ERROR_LIST = []



anti_sound_controller = AntiSoundController(SAMPLESIZE,SAMPLINGRATE,AMOUNT_FREQUENCIES_TO_CANCEL)
anti_sound_controller.start_stream()
anti_sound_controller.main()
anti_sound_controller.stop_stream()