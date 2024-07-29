import configparser as cfp
import os
import numpy as np
import sys

class Param ():
    """ Paramètres """
    def __init__(self, configFile):
        config = cfp.ConfigParser(os.environ)
        config.read(configFile)
        ### PARAMETERS ###
        # Paths
        self.path_data = self.verif_value(config.get('PATHS','input'))[1:-1]
        self.path_results = self.verif_value(config.get('PATHS','output'))[1:-1]
        # Train's and patchs's parameters
        self.nb_epoch = config.getint('VARIABLES', 'epoch')
        self.batch_size = config.getint('VARIABLES', 'batch_train')
        self.batch_map = config.getint('VARIABLES', 'batch_pred')
        self.border = config.getint('DATA', 'border')
        # Model's parameters
        self.pool_layer = self.verif_value(config.get('MODEL', 'pool'))
        self.fusion_layer = self.verif_value(config.get('MODEL', 'fusion'))
        self.drop_rate = self.verif_value(config.get('MODEL', 'dropout'))
        self.nb_filters = self.verif_value(config.get('MODEL', 'filters'))
        self.form_dataset = self.verif_value(config.get('DATA', 'form_dataset'))
        self.cte_loss = config.getfloat('MODEL', 'cte_loss')
        self.specific_form = config.get('DATA', 'specific_form')
        self.optimize = config.get('DATA', 'optmize')
        # Data's parameters
        self.branch1 = config.get('DATA', 'branch_1')
        self.branch2 = config.get('DATA', 'branch_2')
        self.branch3 = config.get('DATA', 'branch_3')
        self.except1 = config.get('DATA', 'except_1')
        self.except2 = config.get('DATA', 'except_2')
        self.except3 = config.get('DATA', 'except_3')
        self.bands = [self.branch1, self.branch2, self.branch3]
        self.excepts = [self.except1, self.except2, self.except3]
        # Will update in preprocess_parameters
        self.nb_branch = 0
        self.type_model = None
        self.list_bands =  None
        self.stack_info =  None
        self.nb_bands = None

        ### DICTIONARIES ###
        self.dict_infosbands = {# MNT SAR (Sentinel-1)
                                "nodata" : [1, "no", "ROI_Stack_MNT_SAR_10m.tif"],
                                "DTM" : [1, "no", "ROI_Stack_MNT_SAR_10m.tif"],
                                "S1avh" : [2, "log", "ROI_Stack_MNT_SAR_10m.tif"],
                                "S1avv" : [3, "log", "ROI_Stack_MNT_SAR_10m.tif"],
                                "S1dvh" : [4, "log", "ROI_Stack_MNT_SAR_10m.tif"],
                                "S1dvv" : [5, "log", "ROI_Stack_MNT_SAR_10m.tif"],
                                # Optique (Sentinel-2)
                                "B02" : [1, "no", "ROI_Stack_Optique_10m.tif"],
                                "B03" : [2, "no", "ROI_Stack_Optique_10m.tif"],
                                "B04" : [3, "no", "ROI_Stack_Optique_10m.tif"],
                                "B05" : [4, "no", "ROI_Stack_MNT_SAR_10m.tif"],
                                "B06" : [5, "no", "ROI_Stack_Optique_10m.tif"],
                                "B07" : [6, "no", "ROI_Stack_Optique_10m.tif"],
                                "B08" : [7, "no", "ROI_Stack_Optique_10m.tif"],
                                "B8A" : [8, "no", "ROI_Stack_Optique_10m.tif"],
                                "B11" : [9, "no", "ROI_Stack_Optique_10m.tif"],
                                "B12" : [10, "no", "ROI_Stack_Optique_10m.tif"],
                                # Derivatives MNT
                                "slope" : [1, "s_min", "max", "no", "with_nodata", "ROI_Stack_derivatives_10m.tif"],
                                "aspect" : [2, "s_min", "max", "no", "with_nodata", "ROI_Stack_derivatives_10m.tif"],
                                "tri01" : [3, "s_min", "max", "no", "with_nodata", "ROI_Stack_derivatives_10m.tif"],
                                "tri05" : [4, "s_min", "max", "no", "with_nodata", "ROI_Stack_derivatives_10m.tif"],
                                "tri10" : [5, "s_min", "max", "no", "with_nodata", "ROI_Stack_derivatives_10m.tif"],
                                "tri15" : [1, "s_min", "max", "no", "with_nodata", "ROI_Stack_derivatives_R15R20_10m.tif"],
                                "tri20" : [2, "s_min", "max", "no", "with_nodata", "ROI_Stack_derivatives_R15R20_10m.tif"],
                                "slrm05" : [6, "s_min", "max", "no", "with_nodata", "ROI_Stack_derivatives_10m.tif"],
                                "slrm10" : [8, "s_min", "max", "no", "with_nodata", "ROI_Stack_derivatives_10m.tif"],
                                "slrm15" : [3, "s_min", "max", "no", "with_nodata", "ROI_Stack_derivatives_R15R20_10m.tif"],
                                "slrm20" : [5, "s_min", "max", "no", "with_nodata", "ROI_Stack_derivatives_R15R20_10m.tif"],
                                "curvtot" : [10, "s_min", "max", "no", "with_nodata", "ROI_Stack_derivatives_10m.tif"],
                                # Information A priori
                                "DistDrainage" : [3, "min", "max", "no", "normale", "ROI_Stack_InfoApriori_Corr_10m.tif"]}

        self.dict_type = {"SAR" : ["SARavh", "SARavv", "SARdvh", "SARdvv"],
                          "Opt" : ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]}

        ### PREPROCESS PARAMETRERS ###
        self.preprocess_parameters()


    def preprocess_parameters(self):
        """
        Function that groups all parameter pre-processing steps
        """
        # Identify the number of model branches
        for band in self.bands :
            if band != "" :
                self.nb_branch += 1

        # Transform dropout rates into a list
        self.drop_rate = self.drop_rate.split(" ")
        self.nb_filters = self.nb_filters.split(" ")

        # Check that the parameters specified in Config.cfg are consistent
        self.verification()

        # Create output file name based on model parameters
        self.create_name_data()

        # Formatting bands and associated information
        self.list_bands, self.nb_bands = self.create_list_bands()
        self.stack_info = self.create_stack_info()

        # Checks that the specified results folder exists
        isExist = os.path.exists(self.path_results)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.path_results)
        

        
    def verification(self):
        """
        Function which checks the data specified by the user in Config.cfg. 
        The following three parts check that :
            - the number of filters and dropouts is consistent with the number of bands specified
            - the names of the specified bands are present in the band dictionary
            - the names of bands to be removed are present in the band group dictionary
        """

        # 1)
        if self.nb_branch != len(self.drop_rate) or self.nb_branch != len(self.nb_filters):
            print("error, the number of filters or dropouts does not correspond to the number of branches in the model")
            sys.exit(1)

        # 2)
        list_bands = "_".join(self.bands)
        list_bands =  list_bands.replace("__", " ").replace("_", " ").split(" ")
        if list_bands[-1] == "":
            list_bands.remove("")
        for band in list_bands:
            test_type = self.dict_type.get(band)
            test_bands = self.dict_infosbands.get(band)
            if test_type == None and test_bands == None:
                print(f"error, the name '{band}' does not exist in the dictionary of the bands")
                sys.exit(1)
            del band, test_type, test_bands

        # 3)
        for i in range (len(self.bands)):
            if self.excepts[i] != "" :
                list_bands = self.bands[i].split(" ")
                for band in list_bands :
                    test = self.dict_type.get(band)
                    if test != None :
                        if self.excepts[i] not in self.dict_type[band] :
                            print(f"error, the band to be removed ({self.excepts[i]}) does not match the specified band group ({band})")
                            sys.exit(1)
                    del band, test
        del list_bands


    def create_list_bands(self): 
        """
        Function to prepare bands for processing in preprocess_data.py
        Band names are listed and the number of bands is counted.
        Adds the "nodata" band to the list to always have the same 
        data-absent zones between different sources
        """

        list_bands = "_".join(self.bands)
        list_bands =  list_bands.replace("__", " ").replace("_", " ").split(" ")
        if list_bands[-1] == "":
            list_bands.remove("")
        list_bands.insert(0, "nodata")

        i = 0
        nb_bands = np.array([0,0,0])
        for branch in self.bands :
            bands_branch = branch.split(" ")
            nb_bands[i] = len(bands_branch)
            if bands_branch[0] == "":
                nb_bands[i] += -1
            for band in bands_branch :
                new_elt = self.dict_type.get(band)
                if new_elt != None :
                    list_bands.remove(band) # and remove the global name
                    nb_bands[i] += -1
                    for elt in new_elt :
                        list_bands.append(elt)
                        nb_bands[i] += 1
                    if self.excepts[i] != "" :
                        list_bands.remove(self.excepts[i]) # remove the name's exception
                        nb_bands[i] += -1
                del new_elt
            i = i+1
            del bands_branch
        nb_bands = np.delete(nb_bands, np.where(nb_bands == 0)[0])
        print(nb_bands)
        return list_bands, nb_bands


    def create_name_data(self):
        """
        Function that creates output file names as follows:
        For one-branch models: Paa_Dbb_Fccc_Xxx_Yyy
            aa = patch size (21, 25, etc.)
            bb = dropout rate (00 for 0.0 and 03 for 0.3)
            ccc = number of filters (008, 032, etc.)
            Xxx = type of last encoder layer (GMP or Fla)
            Yyy = model code (MNT, SAR, Opt)
        For two-branch models: Paa_Dbb_bb_Fccc_ccc_Xxx_Yyy_ZZ_ww
            aa = patch size (21, 25, etc.)
            bb_bb = dropout rate branch 1 and 2 (00_03, 0.0 for branch 1 and 0.3 for branch 2)
            ccc_ccc = number of filters (008_032, 8 for branch 1 and 32 for branch 2)
            Xxx = type of last encoder layer (GMP or Fla)
            Yyy = model code (MNTO, MNTR, OR)
            ZZ = fusion type (LF for Late Fusion and EF for Early Fusion)
            ww = alpha constant value (03 for 0.3 and 05 for 0.5)
        For three-branch models : Paa_Dbb_bb_bb_Fccc_ccc_ccc_Xxx_Yyy_ZZ_ww
            idem
            Yyy = model code (MNTOR)
        """

        self.code_data = self.name_data(' '.join(self.bands))
        name_dropout = str(self.drop_rate[0]).split('.')[1].zfill(2)
        name_filters = str(self.nb_filters[0]).zfill(3)
        for i in range(1, self.nb_branch):
            name_dropout = name_dropout  + "_" + str(self.drop_rate[i]).split('.')[1].zfill(2)
            name_filters = name_filters  + "_" + str(self.nb_filters[i]).zfill(3)
        if self.nb_branch > 1 :
            name_layers = self.pool_layer + self.fusion_layer
            self.type_model = "LF"
        else :
            name_layers = self.pool_layer
            if len(self.bands[0].split(" "))>1:
                self.type_model = "EF"

        self.name = self.name_file(name_dropout, name_filters, name_layers, self.code_data, self.type_model)
        if self.nb_branch == 3:
            self.name = self.name + '_' + str(self.cte_loss).split('.')[1].zfill(2)
        
        del name_dropout, name_filters, name_layers

    def name_data(self, data):
        """ 
        Function defining the model code 
        main codes: MNT, SAR, OPT, MNTO, MNTR, OR, MNTOR
        derivative codes: MNTder, MNTderOR, MNT_Yyy, Opt_SansYyy, etc.
        """

        # 1) DEFINE NAME WITH DATA TYPE
        code = ""

        if "MNT" in data:
            code = code + "MNT"
            
        if "Opt" in data:
            if len(code)>0:
                code = code + "O" # if there are 2 source data
            else :
                code = code + "Opt" # if there is only Opt data

        if "SAR" in data:
            if "Opt" in code :
                code = code[0] + "R" # if there are SAR and Opt data
            elif len(code)>0:
                code = code + "R" # if there is 2 source data
            else :
                code = code + "SAR" # if there is only SAR data
        
        if "Der" in data :
            idx = data.find("T") # find index of "T" letter (case of MNT)
            if idx == -1:
                code = code + "der" # if not add "der" at the end
            else:
                code = code[:idx+1] + "der" + code[idx+1:] # if true : MNTderXxx
            del idx
        
        if "apriori" in data:
            code = code + "info" 
            
        # 2) CASE OF EXCEPTIONS BANDS
        for i in range (len(self.excepts)):
            # if there is an exception, name = NbBands_SansNamebande
            if self.excepts[i] != "":
                name = self.excepts[i].title().split(" ")
                nb_bands = len(self.dict_type[self.bands[i]])
                code = code + "_" + str(nb_bands) + "Sans" + "".join(name)
                del name, nb_bands
        
        # 3) CASE OF SPEFICIQUE/FEW BANDS
        list_word = data.split(" ")
        list_type = "".join(self.dict_type.keys())
        for word in list_word:
            if word in list_type:
                code = code
            elif word == "MNT":
                code = code
            else:
                code = code + "_" + word[0].capitalize() + word[1:]
        del list_word, list_type
        return code


    def name_file(self, dropout, filters, layers, data, type_model=None):
        """ 
        Function that assembles the different parts of the final file name
        """

        if type_model == None:
            name = "_P" + str(self.border*2+1) + "_D" + dropout + "_F" + filters + "_" 
            name =  name + layers + "_" + data
        else :          
            name = "_P" + str(self.border*2+1) + "_D" + dropout + "_F" + filters + "_" 
            name =  name + layers + "_" + data + "_" + type_model
        return name


    def verif_value(self, value):
        """
        Function that checks that every mandatory parameter in Config.cfg is specified
        """

        if len(value) == 0 :
            print("valeur manquante")
            sys.exit(1)
        else :
            return value


    def create_stack_info(self):
        """
        Création du dictionnaire contenant toutes les infos de pré-traitements
        Il sera parcouru dans preprocess_data.py pour qu'à chaque bande, 
        l'algorithme ait le bon nom de fichier pour le charger et la méthode de normalisation 
        """
        
        stack_info = {}
        for name in self.list_bands:
            stack_info[name] = self.dict_infosbands[name]
            del name
        return stack_info



