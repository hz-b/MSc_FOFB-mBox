#DeviceNumbers
#Family
#print " In getBPMFrom IOC"
Calibrationfactor = 0.3051758e-3
BPMIOCData = mml.getpvonline('MDIZ2T5G:bdata')
#BPMIOCData = BPMIOCData[0]
Index = []
Gain  = []
#for i in range(0,len(DeviceList)):
#     Index.append((mml.getfamilydata(Family,'WaveRecordIndex',None,DeviceList[i])-1)[0]);                  
#     Gain.append((mml.getfamilydata(Family,'Monitor','MetaGain',DeviceList[i]))[0]); 
Index = mml.getfamilydata(Family,'WaveRecordIndex',None,idx)-1
Gain = mml.getfamilydata(Family,'Monitor','MetaGain',idx)
Data = BPMIOCData[Index] * Gain * Calibrationfactor;
