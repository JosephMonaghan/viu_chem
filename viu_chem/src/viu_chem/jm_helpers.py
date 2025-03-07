from bs4 import BeautifulSoup, Tag


def reannotate_imzML(receiving_imzML:str,SRC_imzML:str,out_file:str):
    #Error handling for when scan filter extraction fails
    result_file = out_file

    #Retrieve data from source mzml
    with open(SRC_imzML) as file:
        data = file.read()
    data = BeautifulSoup(data,'xml')

    #Open un-annotated imzML
    with open(receiving_imzML) as file:
        data_need_annotation = file.read()
    data_need_annotation = BeautifulSoup(data_need_annotation,'xml')

    ##want to grab and replace all of scansettingslist, instrumentconfigurationlist, grab the filter string, 
    scansettings = data.find("scanSettingsList")
    instrument_config = data.find("instrumentConfigurationList")

    data_need_annotation.find("scanSettingsList").replaceWith(scansettings)
    data_need_annotation.find("instrumentConfigurationList").replaceWith(instrument_config)

    for cvParam in data.select("cvParam"):
        if cvParam["accession"]=="MS:1000512":
            filter_string = cvParam["value"]
    
    for cvParam in data_need_annotation.select("cvParam"):
        if cvParam["accession"]=="MS:1000512":
            cvParam["value"]=filter_string
    
   
    #Write the new file
    with open(result_file,'w') as file:
        file.write(str(data_need_annotation.prettify()))


if __name__=="__main__":
    source_file = "/Users/josephmonaghan/Documents/msi_flow_datatest/msi/HighMR_3A.imzML"
    receiving_file = "/Users/josephmonaghan/Documents/msi_flow_datatest/msi/peakpicking/alignment/matrix_removal/HighMR_3A.imzML"
    reannotate_imzML(receiving_imzML=receiving_file,SRC_imzML=source_file)