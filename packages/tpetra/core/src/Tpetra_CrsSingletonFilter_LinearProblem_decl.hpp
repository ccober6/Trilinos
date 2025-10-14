// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef TPETRA_CRSSINGLETONFILTER_LINEARPROBLEM_DECL_HPP
#define TPETRA_CRSSINGLETONFILTER_LINEARPROBLEM_DECL_HPP

/// \file Tpetra_CrsSingletonFilter_LinearProblem_decl.hpp
/// \brief Declaration of the Tpetra::Transform class

#include "Tpetra_LinearProblem.hpp"
#include "Tpetra_Transform.hpp"

#include "Teuchos_DataAccess.hpp"

#include "Kokkos_Sort.hpp"

#include "Tpetra_Vector_decl.hpp"
#include "Tpetra_MultiVector_decl.hpp"
#include "Tpetra_CrsMatrix_decl.hpp"
#include "Tpetra_RowMatrix_decl.hpp"

namespace Tpetra {

/// \class CrsSingletonFilter_LinearProblem
/// \brief A class for explicitly eliminating matrix rows and columns from a LinearProblem.
///
/// The Epetra_CrsSingletonFilter class takes an existing Epetra_LinearProblem
/// object, analyzes it structure and explicitly eliminates singleton
/// rows and columns from the matrix and appropriately modifies the RHS
/// and LHS of the linear problem.  The result of this process is a
/// reduced system of equations that is itself an Epetra_LinearProblem
/// object.  The reduced system can then be solved using any solver
/// that is understands an Epetra_LinearProblem.  The solution for the
/// full system is obtained by calling ComputeFullSolution().
///
/// Singleton rows are defined to be rows that have a single nonzero
/// entry in the matrix.  The equation associated with this row can be
/// explicitly eliminated because it involved only one variable.  For
/// example if row i has a single nonzero value in column j, call it
/// A(i,j), we can explicitly solve for x(j) = b(i)/A(i,j), where b(i)
/// is the ith entry of the RHS and x(j) is the jth entry of the LHS.
///
/// Singleton columns are defined to be columns that have a single
/// nonzero entry in the matrix.  The variable associated with this
/// column is fully dependent, meaning that the solution for all other
/// variables does not depend on it.  If this entry is A(i,j) then the
/// ith row and jth column can be removed from the system and x(j) can
/// be solved after the solution for all other variables is determined.
///
/// By removing singleton rows and columns, we can often produce a
/// reduced system that is smaller and far less dense, and in general
/// having better numerical properties.
///
/// The basic procedure for using this class is as follows:
/// <ol>
/// <li> Construct full problem: Construct and Epetra_LinearProblem
///      containing the "full" matrix, RHS and LHS.  This is done
///      outside of Epetra_CrsSingletonFilter class.  Presumably,
///      you have some reason to believe that this system may contain
///      singletons.
/// <li> Construct an Epetra_CrsSingletonFilter instance:  Constructor
///      needs no arguments.
/// <li> Analyze matrix: Invoke the Analyze() method, passing in the
///      Tpetra::RowMatrix object from your full linear problem
///      mentioned in the first step above.
/// <li> Go/No Go decision to construct reduced problem:
///      Query the results of the Analyze method using the SingletonsDetected()
///      method.  This method returns "true" if there were singletons
///      found in the matrix.  You can also query any of the other
///      methods in the Filter Statistics section to determine if you
///      want to proceed with the construction of the reduced system.
/// <li> Construct reduced problem:
///      If, in the previous step, you determine that you want to proceed
///      with the construction of the reduced problem, you should next
///      call the ConstructReducedProblem() method, passing in the full
///      linear problem object from the first step.  This method will
///      use the information from the Analyze() method to construct a
///      reduce problem that has explicitly eliminated the singleton
///      rows, solved for the corresponding LHS values and updated the
///      RHS.  This step will also remove singleton columns from the
///      reduced system.  Once the solution of the reduced problem is
///      is computed (via any solver that understands an Epetra_LinearProblem),
///      you should call the ComputeFullSolution() method to compute
///      the LHS values assocaited with the singleton columns.
/// <li> Solve reduced problem: Obtain a RCP to the reduced problem
///      using the ReducedProblem() method.  Using the solver of your
///      choice, solve the reduced system.
/// <li> Compute solution to full problem:  Once the solution the reduced
///      problem is determined, the ComputeFullSolution() method will
///      place the reduced solution values into the appropriate locations
///      of the full solution LHS and then compute the values associated
///      with column singletons.  At this point, you have a complete
///      solution to the original full problem.
/// <li> Solve a subsequent full problem that differs from the original
///      problem only in values: It is often the case that the structure
///      of a problem will be the same for a sequence of linear problems.
///      In this case, the UpdateReducedProblem() method can be useful.
///      After going through the above process one time, if you have a
///      linear problem that is structural \e identical to the previous
///      problem, you can minimize memory and time costs by using the
///      UpdateReducedProblem() method, passing in the subsequent
///      problem.  Once you have called the UpdateReducedProblem()
///      method, you can then solve the reduce problem problem as you
///      wish, and then compute the full solution as before.  The RCP
///      generated by ReducedProblem() will not change when
///      UpdateReducedProblem() is called.
/// </ol>

template <class Scalar,
          class LocalOrdinal,
          class GlobalOrdinal,
          class Node>
class CrsSingletonFilter_LinearProblem : public SameTypeTransform<Tpetra::LinearProblem<Scalar, LocalOrdinal, GlobalOrdinal, Node> > {
 public:
  //! @name Typedefs
  //@{

  using map_type            = Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>;
  using crs_matrix_type     = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using row_matrix_type     = Tpetra::RowMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using multivector_type    = Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using vector_type         = Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using vector_type_int     = Tpetra::Vector<int, LocalOrdinal, GlobalOrdinal, Node>;
  using vector_type_LO      = Tpetra::Vector<LocalOrdinal, LocalOrdinal, GlobalOrdinal, Node>;
  using linear_problem_type = Tpetra::LinearProblem<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using import_type         = Tpetra::Import<LocalOrdinal, GlobalOrdinal, Node>;

  using scalar_type         = Tpetra::Vector<>::scalar_type;
  using local_ordinal_type  = Tpetra::Vector<>::local_ordinal_type;
  using global_ordinal_type = Tpetra::Vector<>::global_ordinal_type;

  using OriginalType      = typename Transform<linear_problem_type, linear_problem_type>::OriginalType;
  using OriginalConstType = typename Transform<linear_problem_type, linear_problem_type>::OriginalConstType;
  using NewType           = typename Transform<linear_problem_type, linear_problem_type>::NewType;
  using NewConstType      = typename Transform<linear_problem_type, linear_problem_type>::NewConstType;

  using nonconst_local_inds_host_view_type  = typename row_matrix_type::nonconst_local_inds_host_view_type;
  using nonconst_global_inds_host_view_type = typename row_matrix_type::nonconst_global_inds_host_view_type;
  using nonconst_values_host_view_type      = typename row_matrix_type::nonconst_values_host_view_type;

  using device_type = typename Node::device_type;
  using vector_view_type_int = Kokkos::View<local_ordinal_type*, device_type>;
  using vector_view_type_scalar = Kokkos::View<scalar_type*, device_type>;
  //@}

  //@{ \name Constructors/Destructor.
  /// \brief Constructor.
  CrsSingletonFilter_LinearProblem(bool verbose = false, bool run_on_host = false);

  /// \brief Destructor
  virtual ~CrsSingletonFilter_LinearProblem() = default;
  //@}

  NewType operator()(const OriginalType& originalLinearProblem);

  void analyze(const OriginalType& originalLinearProblem);

  NewType construct();

  void fwd();

  void rvs();

  //@{ \name Analyze methods.

  /// \brief Analyze the input matrix, removing row/column pairs that have singletons.
  ///
  /// Analyzes the user's input matrix to determine rows and
  /// columns that should be explicitly eliminated to create the
  /// reduced system.  Look for rows and columns that have single
  /// entries.  These rows/columns can easily be removed from the
  /// problem.  The results of calling this method are two
  /// MapColoring objects accessible via RowMapColors() and
  /// ColMapColors() accessor methods.  All rows/columns that
  /// would be eliminated in the reduced system have a color of
  /// 1 in the corresponding RowMapColors/ColMapColors object.
  /// All kept rows/cols have a color of 0.
  void Analyze(const Teuchos::RCP<row_matrix_type>& FullMatrix);

  /// \brief Returns true if singletons were detected in this
  /// matrix (must be called after Analyze() to be effective).
  bool SingletonsDetected() const {
    if (!AnalysisDone_)
      return (false);
    else
      return (NumSingletons() > 0);
  }

  //@}

  //@{ \name Reduce methods.

  /// \brief Return a reduced linear problem based on results of Analyze().
  ///
  /// Creates a new Epetra_LinearProblem object based on the
  /// results of the Analyze phase.  A RCP to the reduced
  /// problem is obtained via a call to ReducedProblem().
  void ConstructReducedProblem(const Teuchos::RCP<linear_problem_type>& Problem);

  /// \brief Update a reduced linear problem using new values.
  ///
  /// Updates an existing Epetra_LinearProblem object using new
  /// matrix, LHS and RHS values.  The matrix structure must be
  /// \e identical to the matrix that was used to construct the
  /// original reduced problem.
  void UpdateReducedProblem(const Teuchos::RCP<linear_problem_type>& Problem);

  //@}

  //@{ \name Methods to construct Full System Solution.

  /// \brief Compute a solution for the full problem using the
  /// solution of the reduced problem, put in LHS of FullProblem().
  ///
  /// After solving the reduced linear system, this method can
  /// be called to compute the solution to the original problem,
  /// assuming the solution for the reduced system is valid. The
  /// solution of the unreduced, original problem will be in the
  /// LHS of the original Epetra_LinearProblem.
  // void ComputeFullSolution();

  //@}

  //@{ \name Filter Statistics.

  /// \brief Return number of rows that contain a single entry,
  /// returns -1 if Analysis not performed yet.
  int NumSingletonRows() const { return (globalNumSingletonRows_); }

  /// \brief Return number of columns that contain a single entry
  /// that are \e not associated with singleton row, returns -1 if
  /// Analysis not performed yet.
  int NumSingletonCols() const { return (globalNumSingletonCols_); }

  /// \brief Return total number of singletons detected.
  ///
  /// Return total number of singletons detected across all
  /// processors.  This method will not return a valid result
  /// until after the Analyze() method is called.  The dimension
  /// of the reduced system can be computed by subtracting this
  /// number from dimension of full system.  \warning This method
  /// returns -1 if Analyze() method has not been called.
  int NumSingletons() const { return (NumSingletonCols() + NumSingletonRows()); }

  /// \brief Returns ratio of reduced system to full system
  /// dimensions, returns -1.0 if reduced problem not constructed.
  double RatioOfDimensions() const { return (RatioOfDimensions_); }

  /// \brief Returns ratio of reduced system to full system nonzero
  /// count, returns -1.0 if reduced problem not constructed.
  double RatioOfNonzeros() const { return (RatioOfNonzeros_); }

  //@}

  //@{ \name Attribute Access Methods.

  /// \brief Returns RCP to the original full (unreduced) Tpetra::LinearProblem.
  Teuchos::RCP<linear_problem_type> FullProblem() const { return (FullProblem_); }

  /// \brief Returns RCP to the derived reduced Tpetra::LinearProblem.
  Teuchos::RCP<linear_problem_type> ReducedProblem() const { return (ReducedProblem_); }

  /// \brief! Returns RCP to Tpetra::RowMatrix from full problem.
  Teuchos::RCP<row_matrix_type> FullMatrix() const { return (FullMatrix_); }

 /// \brief Returns RCP to Tpetra::CrsMatrix from reduced problem.
  Teuchos::RCP<crs_matrix_type> ReducedMatrix() const { return ReducedMatrix_; }

  //! Returns RCP to Tpetra::Map describing the reduced system row distribution.
  Teuchos::RCP<const map_type> ReducedMatrixRowMap() const { return ReducedMatrixRowMap_; }

  //! Returns RCP to Tpetra::Map describing the reduced system column distribution.
  Teuchos::RCP<const map_type> ReducedMatrixColMap() const { return ReducedMatrixColMap_; }

  //! Returns RCP to Tpetra::Map describing the domain map for the reduced system.
  Teuchos::RCP<const map_type> ReducedMatrixDomainMap() const { return ReducedMatrixDomainMap_; }

  //! Returns RCP to Tpetra::Map describing the range map for the reduced system.
  Teuchos::RCP<const map_type> ReducedMatrixRangeMap() const { return ReducedMatrixRangeMap_; }
  //@}

 protected:
  //  This RCP will be null if full matrix is not a CrsMatrix.
  Teuchos::RCP<crs_matrix_type> FullCrsMatrix() const { return (FullCrsMatrix_); }

  Teuchos::RCP<const map_type> FullMatrixRowMap() const { return (FullMatrix()->getRowMap()); }
  Teuchos::RCP<const map_type> FullMatrixColMap() const { return (FullMatrix()->getColMap()); }
  Teuchos::RCP<const map_type> FullMatrixDomainMap() const { return (FullMatrix()->getDomainMap()); }
  Teuchos::RCP<const map_type> FullMatrixRangeMap() const { return (FullMatrix()->getRangeMap()); }

  // int ComputeEliminateMaps();
  int Setup(const Teuchos::RCP<linear_problem_type>& Problem);
  void InitFullMatrixAccess();
  // void GetRow(int localRow, int & NumIndices, int * & Indices);
  void GetRow(local_ordinal_type localRow, size_t& NumIndices,
              Teuchos::Array<local_ordinal_type>& localIndices);
  void GetRow(LocalOrdinal Row, size_t& NumIndices, Teuchos::ArrayView<const Scalar>& Values,
              Teuchos::ArrayView<const LocalOrdinal>& Indices);

  void GetRowGCIDs(LocalOrdinal Row, size_t& NumIndices, Teuchos::ArrayView<const Scalar>& Values,
                   Teuchos::Array<GlobalOrdinal>& GlobalIndices);

  void CreatePostSolveArrays(vector_type_LO localRowIDofSingletonCol,
                             vector_type_LO ColProfiles,
                             vector_type_LO NewColProfiles,
                             vector_type_LO ColHasRowWithSingleton);

  // int ConstructRedistributeExporter(Epetra_Map * SourceMap, Epetra_Map * TargetMap,
  //                                   Epetra_Export * & RedistributeExporter,
  //                                   Epetra_Map * & RedistributeMap);

  Teuchos::RCP<const map_type> GenerateReducedMap(const Teuchos::RCP<const map_type>& originalMap,
                                                  const Teuchos::RCP<vector_type_int>& mapColors,
                                                  int color = 0);

  Teuchos::RCP<linear_problem_type> FullProblem_;
  Teuchos::RCP<linear_problem_type> ReducedProblem_;
  Teuchos::RCP<row_matrix_type> FullMatrix_;
  Teuchos::RCP<crs_matrix_type> FullCrsMatrix_;
  Teuchos::RCP<crs_matrix_type> ReducedMatrix_;
  Teuchos::RCP<multivector_type> ReducedRHS_;
  Teuchos::RCP<multivector_type> ReducedLHS_;

  Teuchos::RCP<const map_type> ReducedMatrixRowMap_;
  Teuchos::RCP<const map_type> ReducedMatrixColMap_;
  Teuchos::RCP<const map_type> ReducedMatrixDomainMap_;
  Teuchos::RCP<const map_type> ReducedMatrixRangeMap_;
  Teuchos::RCP<const map_type> OrigReducedMatrixDomainMap_;
  Teuchos::RCP<import_type> Full2ReducedRHSImporter_;
  Teuchos::RCP<import_type> Full2ReducedLHSImporter_;
  // Teuchos::RCP<export_type> RedistributeDomainExporter_;

  vector_view_type_int ColSingletonRowLIDs_;
  vector_view_type_int ColSingletonColLIDs_;
  vector_view_type_int ColSingletonPivotLIDs_;
  vector_view_type_scalar ColSingletonPivots_;

  local_ordinal_type localNumSingletonRows_;  ///< Number of singleton rows.
  local_ordinal_type localNumSingletonCols_;  ///< Number of singleton columns not eliminated by singleton rows.
  local_ordinal_type globalNumSingletonRows_;
  local_ordinal_type globalNumSingletonCols_;
  double RatioOfDimensions_;
  double RatioOfNonzeros_;

  bool HaveReducedProblem_;
  // bool UserDefinedEliminateMaps_;
  bool AnalysisDone_;
  bool SymmetricElimination_;

  Teuchos::RCP<multivector_type> tempExportX_;
  Teuchos::RCP<multivector_type> tempX_;
  Teuchos::RCP<multivector_type> tempB_;
  // Teuchos::RCP<multivector_type> RedistributeReducedLHS_;

  // Maximum number of entries in any row of the matrix, on this process.
  local_ordinal_type localMaxNumRowEntries_;

  // Array of global indices of length localMaxNumRowEntries_
  Teuchos::ArrayRCP<global_ordinal_type> Indices_;
  // Array of scalars of length localMaxNumRowEntries_
  Teuchos::ArrayRCP<scalar_type> Values_;

  Teuchos::RCP<vector_type_int> RowMapColors_;
  Teuchos::RCP<vector_type_int> ColMapColors_;
  bool FullMatrixIsCrsMatrix_;

  bool run_on_host_;
  bool verbose_;

 private:
  //! Copy constructor (defined as private so it is unavailable to user).
  CrsSingletonFilter_LinearProblem(const Teuchos::RCP<CrsSingletonFilter_LinearProblem>& /* Problem */) {}

 public:

 template<class LocalLO_type, class ColIDView_type, class MapColor_type>
 struct CreatePostSolveArraysFunctor
 {
   LocalLO_type   LocalRowIDofSingletonColData;
   ColIDView_type ColSingletonRowLIDs;
   ColIDView_type ColSingletonColLIDs;

   LocalLO_type  NewColProfilesData;
   LocalLO_type  ColProfilesData;
   LocalLO_type  ColHasRowWithSingletonData;
   MapColor_type ColMapColors_Data;
   MapColor_type RowMapColors_Data;

   CreatePostSolveArraysFunctor(LocalLO_type localRowIDofSingletonColData,
                                ColIDView_type colSingletonRowLIDs, ColIDView_type colSingletonColLIDs,
                                LocalLO_type  newColProfilesData, LocalLO_type colProfilesData, LocalLO_type colHasRowWithSingletonData,
                                MapColor_type colMapColors_Data, MapColor_type rowMapColors_Data) :
   LocalRowIDofSingletonColData(localRowIDofSingletonColData),
   ColSingletonRowLIDs(colSingletonRowLIDs), ColSingletonColLIDs(colSingletonColLIDs),
   NewColProfilesData(newColProfilesData), ColProfilesData(colProfilesData),
   ColHasRowWithSingletonData(colHasRowWithSingletonData),
   ColMapColors_Data(colMapColors_Data), RowMapColors_Data(rowMapColors_Data)
   {}

   KOKKOS_INLINE_FUNCTION
   void operator()(const size_t j, local_ordinal_type &lclNumSingletonCols) const {
     int i = LocalRowIDofSingletonColData(j, 0);
     if (ColProfilesData(j, 0) == 1 && RowMapColors_Data(i, 0) != 1) {
       // These will be sorted by rowLIDs, so no need to be in any particular order
       ColSingletonRowLIDs(lclNumSingletonCols) = i;
       ColSingletonColLIDs(lclNumSingletonCols) = j;
       lclNumSingletonCols ++;
     }
     // Also check for columns that were eliminated implicitly by
     // having all associated row eliminated
     else if (NewColProfilesData(j, 0) == 0 && ColHasRowWithSingletonData(j, 0) != 1 && RowMapColors_Data(i, 0) == 0) {
       ColMapColors_Data(j, 0) = 1;
     }
   }
 };

 template<class Map_type, class LocalPtr_type, class LocalInd_type, class LocalVal_type, class LocalX_type, class LocalB_type,
          class View_type_int, class View_type_scalar, class Err_type>
 struct ConstructReducedProblemFunctor {
   local_ordinal_type NumVectors;
   local_ordinal_type localNumSingletonCols;

   Err_type error_code;
   Map_type lclFullColMap;
   Map_type lclFullRowMap;
   Map_type lclReducedRowMap;

   LocalPtr_type lclFullRowPtr;
   LocalInd_type lclFullColInd;
   LocalVal_type lclFullValues;

   LocalX_type localExportX;
   LocalB_type localRHS;

   View_type_int ColSingletonColLIDs;
   View_type_int ColSingletonRowLIDs;
   View_type_int ColSingletonPivotLIDs;
   View_type_scalar ColSingletonPivots;

   // Constructor
   ConstructReducedProblemFunctor(local_ordinal_type NumVectors_, local_ordinal_type localNumSingletonCols_, Err_type error_code_,
                                  Map_type lclFullColMap_, Map_type lclFullRowMap_, Map_type lclReducedRowMap_,
                                  LocalPtr_type lclFullRowPtr_, LocalInd_type lclFullColInd_, LocalVal_type lclFullValues_,
                                  LocalX_type localExportX_, LocalB_type localRHS_,
                                  View_type_int colSingletonColLIDs_, View_type_int colSingletonRowLIDs_,
                                  View_type_int colSingletonPivotLIDs_, View_type_scalar colSingletonPivots_) :
   NumVectors(NumVectors_), localNumSingletonCols(localNumSingletonCols_), error_code(error_code_),
   lclFullColMap(lclFullColMap_), lclFullRowMap(lclFullRowMap_), lclReducedRowMap(lclReducedRowMap_),
   lclFullRowPtr(lclFullRowPtr_), lclFullColInd(lclFullColInd_), lclFullValues(lclFullValues_),
   localExportX(localExportX_), localRHS(localRHS_),
   ColSingletonColLIDs(colSingletonColLIDs_), ColSingletonRowLIDs(colSingletonRowLIDs_),
   ColSingletonPivotLIDs(colSingletonPivotLIDs_), ColSingletonPivots(colSingletonPivots_)
   {}

   // Functor used in ConstructReducedProblem with parallel-for
   KOKKOS_INLINE_FUNCTION
   void operator()(const local_ordinal_type i) const {
     const LocalOrdinal INVALID = Tpetra::Details::OrdinalTraits<LocalOrdinal>::invalid();
     GlobalOrdinal glbID = lclFullRowMap.getGlobalElement(i);
     LocalOrdinal  lclID = lclReducedRowMap.getLocalElement(glbID);

     if (lclID == INVALID) {
       if (lclFullRowPtr(i+1) == lclFullRowPtr(i)+1) {
         // Singleton row
         LocalOrdinal indX = lclFullColInd(lclFullRowPtr(i));
         Scalar pivot = lclFullValues(lclFullRowPtr(i));
         if (pivot == 0.0) {
           Kokkos::atomic_exchange(&error_code(0), 1);
         } else {
           for (LocalOrdinal j = 0; j < NumVectors; j++) {
             localExportX(indX, j) = localRHS(i, j) / pivot;
           }
         }
       } else {
         // Not singleton row, but need to be removed == Singleton column
         //  look for matching row ID
         int myNum = -1;
         for (local_ordinal_type j = 0; j < localNumSingletonCols; j++) {
           if (ColSingletonRowLIDs[j] == i) {
             myNum = j;
           }
         }
         if (myNum == -1) {
           Kokkos::atomic_exchange(&error_code(0), 3);
         } else {
           //  look for matching col ID
           LocalOrdinal targetCol = ColSingletonColLIDs[myNum];
           for (size_t j = lclFullRowPtr(i); j < lclFullRowPtr(i+1); j++) {
             if (lclFullColMap.getGlobalElement(lclFullColInd[j]) == targetCol) {
               Scalar pivot = lclFullValues[j];
               if (pivot == 0.0) {
                 Kokkos::atomic_exchange(&error_code(0), 2);
               } else {
                 ColSingletonPivotLIDs[myNum] = j;
                 ColSingletonPivots[myNum]    = pivot;
               }
               break;
             }
           }
         }
       }
     }
   }

   // Functor used in UpdateReducedProblem with parallel-reduce
   KOKKOS_INLINE_FUNCTION
   void operator()(const local_ordinal_type i, local_ordinal_type &lclNumSingletonCols) const {
    const LocalOrdinal INVALID = Tpetra::Details::OrdinalTraits<LocalOrdinal>::invalid();
     GlobalOrdinal glbID = lclFullRowMap.getGlobalElement(i);
     LocalOrdinal  lclID = lclReducedRowMap.getLocalElement(glbID);

     if (lclID == INVALID) {
       if (lclFullRowPtr(i+1) == lclFullRowPtr(i)+1) {
         // Singleton row
         LocalOrdinal indX = lclFullColInd(lclFullRowPtr(i));
         Scalar pivot = lclFullValues(lclFullRowPtr(i));
         if (pivot == 0.0) {
           Kokkos::atomic_exchange(&error_code(0), 1);
         } else {
           for (LocalOrdinal j = 0; j < NumVectors; j++) {
             localExportX(indX, j) = localRHS(i, j) / pivot;
           }
         }
       } else {
         // Not singleton row, but need to be removed == Singleton column
         //  look for matching row ID
         int myNum = -1;
         for (local_ordinal_type j = 0; j < localNumSingletonCols; j++) {
           if (ColSingletonRowLIDs[j] == i) {
             myNum =j;
           }
         }
         if (myNum == -1) {
           Kokkos::atomic_exchange(&error_code(0), 3);
         } else {
           //  look for matching col ID
           LocalOrdinal targetCol = ColSingletonColLIDs[myNum];
           for (size_t j = lclFullRowPtr(i); j < lclFullRowPtr(i+1); j++) {
             if (lclFullColMap.getGlobalElement(lclFullColInd[j]) == targetCol) {
               Scalar pivot = lclFullValues[j];
               if (pivot == 0.0) {
                 Kokkos::atomic_exchange(&error_code(0), 2);
               } else {
                 ColSingletonPivots[myNum] = pivot;
               }
               break;
             }
           }
           lclNumSingletonCols++;
         }
       }
     }
   }
 };

};

}  // namespace Tpetra

#endif  //  TPETRA_CRSSINGLETONFILTER_LINEARPROBLEM_DECL_HPP
