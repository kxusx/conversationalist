import av
import io
import numpy as np

class AudioDecoder:
    def __init__(self):
        # Initialize Opus decoder context
        try:
            self.codec = av.codec.CodecContext.create('opus', 'r')
            # self.codec.sample_rate = 16000 # Read-only in newer PyAV
            # self.codec.channels = 1
        except Exception as e:
            print(f"[DECODER] Init error: {e}")
            self.codec = None

    def decode_to_pcm(self, audio_data, codec='pcm'):
        """
        Robustly decode audio to raw PCM (16kHz, 16-bit, Mono).
        """
        if not audio_data:
            return b""
            
        # If already PCM, return as is
        if codec.lower() in ['pcm', 'linear16']:
            return audio_data

        if codec.lower() == 'opus':
            return self._decode_opus_av(audio_data)
        
        return audio_data

    def _decode_opus_av(self, data):
        """Decode Opus using PyAV"""
        if not self.codec:
            return b""
            
        try:
            # Create a packet from the raw data
            packet = av.packet.Packet(data)
            
            # Send packet to decoder
            try:
                self.codec.send_packet(packet)
            except Exception as e:
                # Sometimes send_packet fails on partial/bad data, just ignore
                # print(f"[DECODER] Send error: {e}", flush=True)
                return b""

            pcm_bytes = bytearray()
            
            # Receive all available frames
            while True:
                try:
                    frames = self.codec.receive_frame()
                    if not frames: break # Should not happen if receive_frame returns list, but for AV it yields one usually
                    
                    # Depending on PyAV version, receive_frame returns one frame or raises error when empty
                    # Handling the single frame:
                    frame = frames 
                    
                    # Resample to 16k mono s16le
                    resampler = av.AudioResampler(
                        format='s16',
                        layout='mono',
                        rate=16000
                    )
                    
                    resampled_frames = resampler.resample(frame)
                    for r_frame in resampled_frames:
                        pcm_bytes.extend(r_frame.to_ndarray().tobytes())
                        
                except av.error.EOFError:
                    break
                except av.error.EAGAIN:
                    break
                except Exception as e:
                    # print(f"[DECODER] Receive error: {e}", flush=True)
                    break
            
            return bytes(pcm_bytes)

        except Exception as e:
            # print(f"[DECODER] AV error: {e}", flush=True)
            return b""

