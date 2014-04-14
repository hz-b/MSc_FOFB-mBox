#include "rfm2g_api.h"
#include <iostream>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

#define CTRL_MEMPOS       0x03000000
#define STATUS_MEMPOS     CTRL_MEMPOS + 50
#define MESSAGE_MEMPOS       CTRL_MEMPOS + 100
#define CONFIG_MEMPOS        CTRL_MEMPOS + 1000

RFM2GHANDLE    RFM_Handle = 0;
RFM2G_NODE     NodeId;
char   devicename[] =    "/dev/rfm2g0";
char   result          = 0;
using namespace std;

void init_rfm() {
    cout << "Init RFM" << endl;
    result = RFM2gOpen( devicename, &RFM_Handle );
    if (result) {
        cout << "  Can't open " << devicename << "\n" << endl;
        exit(1);
    }
    cout << "  RFM Handle : " << RFM_Handle << endl;
    result = RFM2gNodeID (RFM_Handle, &NodeId);
    if (result) {
        cout << "  Can't get Node Id" << endl;
        exit(1);
    }
    cout << "  RFM Node Id : " << NodeId << endl;
}

void sendMessage(const char* Message,const char *error) {
   cout << "Send Message: " << Message << " Error: " << error << endl;
   unsigned long pos = MESSAGE_MEMPOS;
   cout << "Send To Pos: " << pos << endl;
   struct t_header {
        unsigned short namesize;
        unsigned short sizey;
        unsigned short  sizex;
        unsigned short type;
   } header;
   int thesize = 2 + sizeof(header)+ 6 + strlen(Message) +
                     sizeof(header)+ 5 + strlen(error) ;
   unsigned char * mymem = (unsigned char *) malloc(thesize);
   unsigned long structpos = 0;

   mymem[0]=2;  mymem[1] = 0; structpos += 2;// number of Elements (message, error)
    
   header.namesize=6;
   header.sizex = strlen(Message);
   header.sizey = 1;
   header.type = 2;  
   memcpy(mymem+structpos,&header,sizeof(header)); structpos += sizeof(header);
   memcpy(mymem+structpos,"status",6); structpos += 6;
   memcpy(mymem+structpos,Message,strlen(Message)); structpos += strlen(Message);

   header.namesize=5;
   header.sizex = strlen(error);
   header.sizey = 1;
   header.type = 2; 
   memcpy(mymem+structpos,&header,sizeof(header)); structpos += sizeof(header);
   memcpy(mymem+structpos,"error",5); structpos += 5;
   memcpy(mymem+structpos,error,strlen(error)); structpos += strlen(error); 
   
   result = RFM2gWrite( RFM_Handle, pos , mymem, thesize); 
   //unsigned short l = 2;
   //result = RFM2gWrite( RFM_Handle, pos , &l, 2); 
   cout << "Result" << result << endl;
   free(mymem);
}

int main() {
   init_rfm();   
   sendMessage("TEST MESSAGE2","TEST ERROR2");
   return 0;
}
