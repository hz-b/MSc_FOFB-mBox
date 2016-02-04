#include "rfm_helper.h"

void RFMHelper::sendMessage(const char* Message, const char *error)
{
    unsigned long pos = MESSAGE_MEMPOS;
    //cout << "Send To Pos: " << pos << endl;
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

    m_driver->write(pos , mymem, thesize);
    //unsigned short l = 2;
    //result = RFM2gWrite( RFM_Handle, pos , &l, 2); 
    //cout << "Result" << result << endl;
    free(mymem);
}

void RFMHelper::dumpMemory(void* data, int len) {
    unsigned char* p = (unsigned char*)data;
    std::printf("'%f'\n", *(double*)p);
    for (int i = 0 ; i < len ; ++i) {
        std::printf("0x%x\n", (unsigned char)*(p+i));
    }
}

void RFMHelper::dumpMemory(volatile void* data, int len) {
    unsigned char* p = (unsigned char*)data;
    std::printf("'%f'\n", *(double*)p);
    for (int i = 0 ; i < len ; ++i) {
        std::printf("0x%x\n", (unsigned char)*(p+i));
    }
}

void RFMHelper::prepareField(arma::vec& field, unsigned long pos, unsigned long dim1, unsigned long dim2) {
    RFM2G_UINT32 threshold = 0, data_size=dim1*dim2*8;
    /* see if DMA threshold and buffer are intialized */
    m_driver->getDMAThreshold( &threshold );

    if (data_size<threshold) {
        // use PIO  tranfer
        field.set_size(dim1*dim2);
        m_driver->read(pos,(void*)field.memptr(), data_size);
        dumpMemory(field.memptr(),8);
    } else {
        // use DMA transfer
        m_driver->read(pos,(void*)m_dma->memory(), data_size);
        std::cout << "vec DMA " << std::endl;
        dumpMemory(m_dma->memory(),8);
        field = arma::vec((const double *)m_dma->memory(),dim1*dim2);
    }
    //cout << "v5" << field << endl;
    //cout<<"read vec at pos "<<pos<<" raw len:"<<data_size<<" eff. len: " << data_size/8 << endl;
    char* p=(char*)field.memptr();
    //printf("'%x'\n",(RFM2G_UINT64)(*(pDmaMemory+pos)));
    //cout<<field<<endl;
}

void RFMHelper::prepareField(arma::mat& field, unsigned long pos, unsigned long dim1, unsigned long dim2)
{
    RFM2G_UINT32 threshold = 0, data_size = dim1*dim2*8;
    /* see if DMA threshold and buffer are intialized */
    m_driver->getDMAThreshold( &threshold );

    std::cout << "pos="<<std::setw(12) << pos << std::endl;

    if (data_size < threshold) {
        // use PIO transfer
        field.set_size(dim1,dim2);
        m_driver->read(pos, (void*) field.memptr(), data_size);
    } else {
        // use DMA transfer
        m_driver->read(pos, (void*) m_dma->memory(), data_size);
        dumpMemory(m_dma->memory(),8);
        dumpMemory(m_dma->memory()+8,8);
        dumpMemory(m_dma->memory()+16,8);

        field = arma::mat((const double *) m_dma->memory(), dim1, dim2);
    }
    std::cout << "m5" << field(0,0) << " " << field(0,1) << std::endl;
    std::cout << "Size : " << field.n_cols << ":"<< field.n_rows << std::endl;
}

void RFMHelper::searchField(std::string &name,
                            unsigned long &pos,
                            unsigned long &datasize1,
                            unsigned long &datasize2,
                            unsigned long &datasize
                           )
{
        short header[4];
        m_driver->read(pos, &header, 8); // 4 * 16-Bit = 8
        pos += 8;

        unsigned long namesize = header[0];
        datasize1 = header[1];
        datasize2 = header[2];
        datasize  = datasize1 * datasize2;
        unsigned long type  = header[3];
        if (type == 1)
            datasize *= 8;

        char name_tmp[namesize];
        m_driver->read(pos, name_tmp, namesize);
        name_tmp[namesize] = '\0';
        name = std::string(name_tmp);

        pos += namesize;
        std::cout << "name:" << " "<< name << std::endl;
}