class active_hours_class:
#    def __init__(self, date_time, s):
    def active_hours(self,records):
        newList=[]
        dateList=[]

    #current_time = newList.strftime("%H:%M:%S")
        import datetime 
        
        day_name= ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']
        for row in records:
            #print("Hello________")
            #print(row[1])
            #print(type(row[7]))
            z= row
            z=z[7] ##cHANGE this to index 7 as the datetime field returned from the query has the index for create_date=query object[7]
            current_time = z.strftime("%H:%M:%S")
            current_date=z.strftime('%d %m 20%y')
            x = current_time.split(":", 2)
            #print("X =" + str(x))
            #print("date: " + str(current_date))
            day = datetime.datetime.strptime(current_date, '%d %m %Y').weekday()
            #print(type(day))
            #x=x[1]
            #x=x.split(":")
            dateList.append(day)
            x="".join(x)
            newList.append(x)
        #print(dateList)
        #print("lenght of datelist" + str(len(dateList)))    
        #print(newList)
        #print(len(newList))

        day_zeros=[0]*7
        hour=[0]*24
        ran=range(1,250000,10000)
        hour_check=range(0,24,1)
        print(hour_check)
        day_check=range(0,7,1)
        #index=0
        #print(hour)
        ######To calculate how average of times per day basis#### WEEKLY basis
        #for that_day in dateList:
        #    for particular_day in day_check:
        #        if that_day==particular_day:
        #            day_zeros[that_day]=day_zeros[that_day]+1
        #print(day_zeros)
                #if that1 in dateList[particular_day]:
                    #day_zeros[that1-1]=day_zeros[that1-1]+1
                    #print(day_zeros)  
        #####################################Weekly  basis            
        import numpy as np
        date_time_array = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]*7, np.int32)
        #print(new_vari.shape)
        count=0         
        for i in newList:
            that_day=dateList[count]    
            for particular_day in day_check:
                if that_day==particular_day:
                    day_zeros[that_day]=day_zeros[that_day]+1
            that_time=newList[count]
            for itere in hour_check:
                #print(i)
                i=int(i)
                #print(index)
                #index=index + 1
                #print('ranfs',ran[itere])
                that_time=int(that_time)
                if that_time in range(ran[itere-1], ran[itere]):
                    #print(i)
                    #print(ran[itere-1], ran[itere])
                    #print(index)
                    #print(itere)
                    hour[itere-1]= hour[itere-1]+ 1 
                    date_time_array[that_day][itere-1]=date_time_array[that_day][itere-1]+1      
            count=count+1     
        print("\n Average numbr of users in 24 Hours range= ", hour)
        print("\nAverage on Weekly basis: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']\n",day_zeros)
        print("\n Active users in 7 days x 24 hours: \n",date_time_array)
        return hour
