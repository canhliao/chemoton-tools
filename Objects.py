import scine_database as db
import scine_database.energy_query_functions as dbfxn
import scine_molassembler as masm
import scine_chemoton.gears.pathfinder as pf
import numpy as np

class Model(db.Model):
        def __init__(self, method_family: str, method: str, basisset: str, spin_mode: str, program: str):
                super().__init__(method_family, method, basisset)
                self.spin_mode = spin_mode
                self.program = program


class DatabaseManager(db.Manager):
        def __init__(self, db_name: str, ip: str, port: int):
                super().__init__()

                credentials = db.Credentials(ip, port, db_name)
                self.set_credentials(credentials)
                self.connect()

                self.reaction_collection_ = None
                self.compound_collection_ = None
                self.structure_collection_ = None
                self.properties_collection_ = None
                self.elementary_step_collection_ = None

        def collectReactions(self):
                self.reaction_collection_ = self.get_collection("reactions")

        def collectCompounds(self):
                self.compound_collection_ = self.get_collection("compounds")

        def collectStructures(self):
                self.structure_collection_ = self.get_collection("structures")

        def collectProperties(self):
                self.properties_collection_ = self.get_collection("properties")

        def collectElementarySteps(self):
                self.elementary_step_collection_ = self.get_collection("elementary_steps")

        def loadCollections(self):
                self.collectReactions()
                self.collectCompounds()
                self.collectStructures()
                self.collectProperties()
                self.collectElementarySteps()


class Finder(pf.Pathfinder):
        def __init__(self, manager: DatabaseManager):
                super().__init__(manager)

        def loadGraph(self):
                self.options.graph_handler = "basic"
                self.build_graph()
                
 
class Reaction():
        def __init__(self, energyType: str):
                self.id_ = None; # database id object
                self.dbReaction_ = None # database reaction object
                self.energyType_ = energyType # "electronic_energy" or "gibbs_free_energy"

                self.Er_ = 0.0 # energy of reactants
                self.Ep_ = 0.0 # energy of products
                self.fwdBarrier_ = 0.0 # energy barrier of forward reaction
                self.bwdBarrier_ = 0.0 # energy barrier of backward reaction
                self.fwdK_ = 0.0 # equilibrium constant for forward reaction 
                self.bwdK_ = 0.0 # equilibrium constant for backward reaction
                self.minimum_energy_elementary_step_ = None # database elementary step object

                self.reactantStructures_ = [] # list of database id objects for structures
                self.productStructures_ = [] # list of database id objects for structures
                self.reactants_ = [] # list of database id objects for reactant compounds or flasks
                self.products_ = [] # list of database id objects for product compounds or flasks

                self.isDefective_ = False # flags something wrong with reaction

        def setObjectWithObject(self, dbReaction: db.Reaction):
                self.dbReaction_ = dbReaction
                self.id_ = dbReaction.get_id()

        def setObjectWithId(self, idstring: str, manager: DatabaseManager):
                self.id_ = db.ID(idstring)
                self.dbReaction_ = db.Reaction(self.id_, manager.reaction_collection_)

        def loadReactantsAndProducts(self, manager: DatabaseManager, model: Model):
                if self.minimum_energy_elementary_step_ is None:
                        raise ValueError("Elementary Step not loaded")
                self.reactants_, self.products_ = self.dbReaction_.get_reactants(db.Side.BOTH)
                reactantTypes, productTypes = self.dbReaction_.get_reactant_types(db.Side.BOTH)
                # We are neglecting all flasks
                for rtype in reactantTypes:
                        self.isDefective_ = self.isDefective_ or (rtype == db.CompoundOrFlask.FLASK)
                for ptype in productTypes:
                        self.isDefective_ = self.isDefective_ or (ptype == db.CompoundOrFlask.FLASK)

                self.reactantStructures_, self.productStructures_ = self.minimum_energy_elementary_step_.get_reactants(db.Side.BOTH)
                self.Er_ = 0.0
                for r_id in self.reactantStructures_:
                        try:
                                self.Er_ += dbfxn.get_energy_for_structure(db.Structure(r_id), self.energyType_, model, manager.structure_collection_, manager.properties_collection_)
                        except:
                                print(r_id.string())

                self.Ep_ = 0.0
                for p_id in self.productStructures_:
                        self.Ep_ += dbfxn.get_energy_for_structure(db.Structure(p_id), self.energyType_, model, manager.structure_collection_, manager.properties_collection_)

        def calculateEqConstant(self, temp: float):
                self.fwdK_ = np.exp((self.Ep_ - self.Er_) / (1.9071 * temp))
                if self.fwdK_ < 1.0e-8:
                        raise ValueError("Forward equilibrium constants is near zero.")
                self.bwdK_ = 1.0 / self.fwdK_

        def loadMinEnergyElementaryStep(self, manager: DatabaseManager, model: Model):
                self.minimum_energy_elementary_step_ = dbfxn.get_elementary_step_with_min_ts_energy(self.dbReaction_, self.energyType_, model, manager.elementary_step_collection_, manager.structure_collection_, manager.properties_collection_)

        def loadBarriers(self, manager: DatabaseManager, model: Model):
                if self.minimum_energy_elementary_step_ is None:
                        raise ValueError("Elementary Step not loaded")
                self.fwdBarrier_, self.bwdBarrier_ = dbfxn.get_barriers_for_elementary_step_by_type(self.minimum_energy_elementary_step_, self.energyType_, model, manager.structure_collection_, manager.properties_collection_)

        def lookForDefects(self):
                if self.fwdK_ < 0.0 or self.fwdBarrier_ < 0.0 or self.bwdBarrier_ < 0.0:
                        self.isDefective_ = True


class ReactionIterator():
        def __init__(self, manager: DatabaseManager):
                self.manager_ = manager
                self.iter_ = self.manager_.reaction_collection_.iterate_all_reactions()

        def __iter__(self):
                return self

        def __next__(self) -> db.Reaction:
                rxn = next(self.iter_)
                rxn.link(self.manager_.reaction_collection_)
                return rxn

        def reset(self):
                self.iter_ = self.manager_.reaction_collection_


class Compound():
        def __init__(self):
                self.smiles_ = ""
                self.id_ = None
                self.dbCompound_ = None
                self.multiplicity_ = 0
                self.isFeasible_ = True
                self.pathway_list = [] # list of SynthesisPathway objects, one for each plausible starting compound


        def setObjectWithObject(self, dbCompound: db.Compound):
                self.dbCompound_ = dbCompound
                self.id_ = dbCompound.get_id()

        def setObjectWithId(self, idstring: str, manager: DatabaseManager):
                self.id_ = db.ID(idstring)
                self.dbCompound_ = db.Compound(self.id_, manager.compound_collection_)

        def loadSMILES(self, manager: DatabaseManager):
                struct = self.dbCompound_.get_structures(manager)[0]
                cbor = struct.get_graph("masm_cbor_graph")
                cbor_bin = masm.JsonSerialization.base_64_decode(cbor)
                serializer = masm.JsonSerialization(cbor_bin, masm.JsonSerialization.BinaryFormat.CBOR)
                mol = serializer.to_molecule()
                try:
                        self.smiles_ = masm.io.experimental.emit_smiles(mol)
                except Exception as e:
                        self.smiles_ = "None"

        def loadMultiplicity(self, manager: DatabaseManager):
                centroid = db.Structure(self.dbCompound_.get_centroid(), manager.structure_collection_)
                self.multiplicity_ = centroid.get_multiplicity()

        def findPathToCompound(self, manager: DatabaseManager, finder: Finder, model: Model, startingCompoundIdStrings: list[str], maxBarrier: float, minEqConstant: float, temp: float):
                for c in startingCompoundIdStrings:
                        message = "Searching for reaction pathways from " + c + " to " + self.id_.string()
                        print(message)
                        path_from_c = SynthesisPathway(manager, finder, model, c, self.id_.string())
                        try:
                                path_from_c.find(maxBarrier, minEqConstant, temp, 10)
                        except:
                                message = "Failed to locate compound " + self.id_.string() + " in reaction graph."
                                print(message)
                                continue
                        if len(path_from_c.reaction_pathways) > 0:
                                self.pathway_list.append(path_from_c)

                if self.pathway_list == []:
                        self.isFeasible_ = False
                        

class CompoundIterator():
        def __init__(self, manager: DatabaseManager):
                self.manager_ = manager
                self.iter_ = self.manager_.compound_collection_.iterate_all_compounds()

        def __iter__(self):
                return self

        def __next__(self) -> db.Compound:
                compound = next(self.iter_)
                compound.link(self.manager_.compound_collection_)
                return compound

        def reset(self):
                self.iter_ = self.manager_.compound_collection_.iterate_all_compounds()
                
                               
class SynthesisPathway():
        def __init__(self, manager: DatabaseManager, finder: Finder, model: Model, startingCompoundIdString: str, targetCompoundIdString: str):
                self.manager_ = manager
                self.finder_ = finder
                self.model_ = model
                self.startingCompoundIdString_ = startingCompoundIdString
                self.targetCompoundIdString_ = targetCompoundIdString

                # list of reaction network graph node ids
                # starts with compound id of the starting compound and ends with target compound
                # even elements are compound id string
                # odd elements are reaction id strings appended with ";n;"
                # where n = 0 means the compounds are connected by the "forward" reaction 
                # and n = 1 means the "backward" reaction
                self.reaction_pathways = [] # list of reaction id strings
                self.compound_pathways = [] # list of compound id strings

        def find(self, maxBarrier: float, minEqConstant: float, temp: float, nPaths: int):
                try:
                        paths = self.finder_.find_paths(self.startingCompoundIdString_, self.targetCompoundIdString_, 10)
                except:
                        raise ValueError("Node not found in graph.")
                for p in paths:
                        if len(p[0]) < 2:
                                continue

                        rxnPath = p[0][1::2]
                        compoundPath = p[0][::2]

                        include = True
                        for r in rxnPath:
                                r_id, rORp, _ = r.split(";")
                                rxn = Reaction("electronic_energy")
                                rxn.setObjectWithId(r_id, self.manager_)
                                try:
                                        rxn.loadMinEnergyElementaryStep(self.manager_, self.model_)
                                        rxn.loadReactantsAndProducts(self.manager_, self.model_)
                                        rxn.loadBarriers(self.manager_, self.model_)
                                        rxn.calculateEqConstant(temp)
                                        rxn.lookForDefects()
                                except:

                                        include = False
                                        print("Failed Pathway 1")
                                        break

                                if rxn.isDefective_:
                                        include = False
                                        break

                                barrier = 0.0
                                k = 0.0
                                if rORp == "0":
                                        barrier = rxn.fwdBarrier_
                                        k = rxn.fwdK_
                                elif rORp == "1":
                                        barrier = rxn.bwdBarrier_
                                        k = rxn.bwdK_
                                else:
                                        include = False
                                        break

                                if barrier > maxBarrier or k < minEqConstant:
                                        include = False
                        if not include:
                                continue

                        first_rxn_id, first_rORp, _ = rxnPath[0].split(";")
                        last_rxn_id, last_rORp, _ = rxnPath[-1].split(";")
                        first_rxn = Reaction("electronic_energy")
                        last_rxn = Reaction("electronic_energy")
                        first_rxn.setObjectWithId(first_rxn_id, self.manager_)
                        last_rxn.setObjectWithId(last_rxn_id, self.manager_)

                        first_rxn.loadMinEnergyElementaryStep(self.manager_, self.model_)
                        first_rxn.loadReactantsAndProducts(self.manager_, self.model_)
                        last_rxn.loadMinEnergyElementaryStep(self.manager_, self.model_)
                        last_rxn.loadReactantsAndProducts(self.manager_, self.model_)

                        reactant_energy = 0.0
                        product_energy = 0.0
                        if first_rORp == "0":
                                reactant_energy = first_rxn.Er_
                        elif first_rORp == "1":
                                reactant_energy = first_rxn.Ep_
                        else:
                                continue

                        if last_rORp == "0":
                                product_energy = last_rxn.Ep_
                        elif first_rORp == "1":
                                product_energy = last_rxn.Er_
                        else:
                                continue

                        total_k = np.exp((product_energy - reactant_energy) / (1.9071 * temp))

                        if total_k < minEqConstant:
                                include = False
                        
                        if include:
                                self.reaction_pathways.append(rxnPath)
                                self.compound_pathways.append(compoundPath)

if __name__ == "__main__":
        db_name = "ch3sh-ch2sh"
        ip = "localhost"
        port = 27017

        startingCompoundIdList = []

        manager = DatabaseManager(db_name, ip, port)
        manager.loadCollections()
        model = Model('dft', 'm062x', '6-311+G**', "unrestricted", "orca")
        maxBarrier = 150 # kJ/mol
        minK = 0.01
        temp = 300 # K
        approvedMultiplicity = [1, 3]

        finder = Finder(manager)
        finder.loadGraph()

        starting_compound_id_strings = ["69c1abfbdf7e55117102846a", "69c1abfbdf7e55117102846a", \
                                        "69c290cd54afd82e0701e3f3", "69c2a6b754afd82e0701e3ff", \
                                        "69c2e5d854afd82e0701e43d", "69c2f41054afd82e0701e449", \
                                        "69c2f7da54afd82e0701e459"]

        compounds_id_strings = []
        compound_iter = CompoundIterator(manager)
        print("Deciding which compounds are feasible.")
        for c in compound_iter:
                c_obj = Compound()
                c_obj.setObjectWithObject(c)
                c_obj.loadMultiplicity(manager)
                if c_obj.multiplicity_ not in approvedMultiplicity:
                        continue
                c_obj.findPathToCompound(manager, finder, model, starting_compound_id_strings, \
                                         maxBarrier, minK, temp)
                if c_obj.isFeasible_:
                        compounds_id_strings.append(c_obj.id_.string())

        for c in starting_compound_id_strings:
                if c not in compounds_id_strings:
                        compounds_id_strings.append(c)

        with open("molecules.txt", "a") as file:
                file.write("Feasible molecules:\n")
                for c in compounds_id_strings:
                        c_obj = Compound()
                        c_obj.setObjectWithId(c, manager)
                        c_obj.loadMultiplicity(manager)
                        c_obj.loadSMILES(manager)
                        written_string = c + "," + c_obj.smiles_ + "," + str(c_obj.multiplicity_) + "\n"
                        print(written_string)
                        file.write(written_string)

        reactions = []
        reaction_iter = ReactionIterator(manager)
        print("Deciding which reactions are feasible.")
        for r in reaction_iter:
                r_obj = Reaction("electronic_energy")
                r_obj.setObjectWithObject(r)
                r_obj.loadMinEnergyElementaryStep(manager, model)
                r_obj.loadReactantsAndProducts(manager, model)
                r_obj.calculateEqConstant(temp)
                r_obj.loadBarriers(manager, model)
                r_obj.lookForDefects()

                if r_obj.isDefective_:
                        continue

                # check forward reaction
                if r_obj.fwdBarrier_ < maxBarrier and r_obj.fwdK_ > minK:
                        reactant_smiles = []
                        product_smiles = []
                        allowRxn = True
                        for reactant_id in r_obj.reactants_:
                                # reactants must be in the list of feasible molecules
                                if reactant_id.string() not in compounds_id_strings:
                                        allowRxn = False
                                        break
                                c_obj = Compound()
                                c_obj.setObjectWithId(reactant_id.string(), manager)
                                c_obj.loadSMILES(manager)
                                reactant_smiles.append(c_obj.smiles_)
                                # if SMILES fail
                                if c_obj.smiles_ == "" or c_obj.smiles_ == reactant_id.string():
                                        allowRxn = False
                                        break

                        for product_id in r_obj.products_:
                                if not allowRxn:
                                        break
                                c_obj = Compound()
                                c_obj.setObjectWithId(product_id.string(), manager)
                                c_obj.loadSMILES(manager)
                                product_smiles.append(c_obj.smiles_)
                                # if SMILES fail
                                if c_obj.smiles_ == "" or c_obj.smiles_ == product_id.string():
                                        allowRxn = True
                                        break
                        if allowRxn:
                                rxn_string = ""
                                for smiles in reactant_smiles:
                                        rxn_string += smiles + " + "
                                rxn_string = rxnstring[:-2] + "-> "
                                for smiles in product_smiles:
                                        rxn_string += smiles + " + "
                                rxn_string = rxnstring[:-2]
                                reactions.append((r_obj.string() + ";f;", rxn_string, r_obj.fwdBarrier_, r_obj.fwdK_))
                        
                # check backward reaction
                if r_obj.bwdBarrier_ < maxBarrier and r_obj.bwdK_ > minK:
                        reactant_smiles = []
                        product_smiles = []
                        allowRxn = True
                        for reactant_id in r_obj.products_:
                                # reactants must be in the list of feasible molecules
                                if reactant_id.string() not in compounds_id_strings:
                                        allowRxn = False
                                        break
                                c_obj = Compound()
                                c_obj.setObjectWithId(reactant_id.string(), manager)
                                c_obj.loadSMILES(manager)
                                reactant_smiles.append(c_obj.smiles_)
                                # if SMILES fail
                                if c_obj.smiles_ == "" or c_obj.smiles_ == reactant_id.string():
                                        allowRxn = False
                                        break

                        for product_id in r_obj.reactants_:
                                if not allowRxn:
                                        break
                                c_obj = Compound()
                                c_obj.setObjectWithId(product_id.string(), manager)
                                c_obj.loadSMILES(manager)
                                product_smiles.append(c_obj.smiles_)
                                # if SMILES fail
                                if c_obj.smiles_ == "" or c_obj.smiles_ == product_id.string():
                                        allowRxn = False
                                        break
                        if allowRxn:
                                rxn_string = ""
                                for smiles in reactant_smiles:
                                        rxn_string += smiles + " + "
                                rxn_string = rxn_string[:-2] + "-> "
                                for smiles in product_smiles:
                                        rxn_string += smiles + " + "
                                rxn_string = rxn_string[:-2]
                                reactions.append((r_obj.id_.string() + ";b;", rxn_string, r_obj.bwdBarrier_, r_obj.bwdK_))

        with open("reactions.txt", "a") as file:
                file.write("ReactionId, Reaction, Barrier (kJ/mol)\n")
                for r in reactions:
                        written_string = r[0] + "," + r[1] + "," + str(r[2]) + "," + str(r[3]) + "\n"
                        file.write(written_string)
                        print(r)
                        












