from active_hours_file import active_hours_class
import mysql.connector
from mysql.connector import errorcode
from datetime import datetime
class active_hrs_class_main_call:
    def active_function_main(self):
        try:
            cnx = mysql.connector.connect(user='root', password='123456',
                                            database='tickets')
            sql_select_Query = "select * from ticket_table_1"
            cursor = cnx.cursor()
            cursor.execute(sql_select_Query)
            records = cursor.fetchall()
            print("Total number of rows in Tickets_table is: ", cursor.rowcount)  
            print("\nPrinting each ticket record")
            for row in records:
                print(row)
                #   # print("Id = ", row[0])
                #    #print("Platform_parameter_id = ", row[1])
                #    #print("company  = ", row[2])
                #    #print("user_id  = ", row[3], "\n")
                #print("end_user_id  = ", row[4])
                #   # print("end_user_name = ", row[5])
                # print("ticket_status_id= ", row[6])
                #    print("created_ date= ", row[7])
                #    print("end_user_info= ", row[8])
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with your user name or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database does not exist")
            else:
                print(err)
        else:
            print("Close")
        cnx.close()
        #print(type(records))
        methodvar=active_hours_class()
        methodvar.active_hours(records)

meth=active_hrs_class_main_call()
meth.active_function_main()