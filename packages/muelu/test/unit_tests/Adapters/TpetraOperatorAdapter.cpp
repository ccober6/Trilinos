// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_ScalarTraits.hpp>

#include <MueLu_TestHelpers.hpp>
#include <MueLu_Version.hpp>

#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_TpetraMultiVector.hpp>

#include <MueLu_FactoryManagerBase.hpp>
#include <MueLu_Hierarchy.hpp>
#include <MueLu_PFactory.hpp>
#include <MueLu_SaPFactory.hpp>
#include <MueLu_TransPFactory.hpp>
#include <MueLu_RAPFactory.hpp>
#include <MueLu_AmesosSmoother.hpp>
#include <MueLu_TrilinosSmoother.hpp>
#include <MueLu_SmootherFactory.hpp>
#include <MueLu_TentativePFactory.hpp>
#include <MueLu_AmesosSmoother.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>

namespace MueLuTests {

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(TpetraOperator, Apply, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include <MueLu_UseShortNames.hpp>
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  out << "version: " << MueLu::Version() << std::endl;

  typedef MueLu::TpetraOperator<SC, LO, GO, NO> muelu_tpetra_operator_type;
  typedef typename Teuchos::ScalarTraits<SC>::magnitudeType magnitude_type;

  if (TestHelpers::Parameters::getLib() == Xpetra::UseTpetra) {
    // matrix
    RCP<const Teuchos::Comm<int>> comm = TestHelpers::Parameters::getDefaultComm();
    RCP<Matrix> Op                     = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(6561 * comm->getSize());  //=8*3^6
    RCP<const Map> map                 = Op->getRowMap();

    RCP<MultiVector> nullSpace = MultiVectorFactory::Build(map, 1);
    nullSpace->putScalar((SC)1.0);
    // Teuchos::Array<Teuchos::ScalarTraits<SC>::magnitudeType> norms(1);
    Teuchos::Array<magnitude_type> norms(1);
    nullSpace->norm1(norms);

    RCP<Hierarchy> H = rcp(new Hierarchy());
    H->setDefaultVerbLevel(Teuchos::VERB_NONE);

    RCP<MueLu::Level> Finest = H->GetLevel();
    Finest->setDefaultVerbLevel(Teuchos::VERB_NONE);
    Finest->Set("A", Op);
    H->Setup();

    // ------------- test Tpetra Operator wrapping MueLu hierarchy ------------
    RCP<muelu_tpetra_operator_type> tH = rcp(new muelu_tpetra_operator_type(H));

    RCP<MultiVector> RHS1 = MultiVectorFactory::Build(Op->getRowMap(), 1);
    RCP<MultiVector> X1   = MultiVectorFactory::Build(Op->getRowMap(), 1);

    // normalized RHS, zero initial guess
    RHS1->setSeed(846930886);
    RHS1->randomize();
    RHS1->norm2(norms);
    RHS1->scale(1 / norms[0]);

    X1->putScalar((SC)0.0);

    tH->apply(*(Xpetra::toTpetra(RHS1)), *(Xpetra::toTpetra(X1)));

    X1->norm2(norms);
    out << "after apply, ||X1|| = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) << norms[0] << std::endl;

    // -------------- test MueLu Hierarchy directly -----------------------
    RCP<MultiVector> RHS2 = MultiVectorFactory::Build(Op->getRowMap(), 1);
    RCP<MultiVector> X2   = MultiVectorFactory::Build(Op->getRowMap(), 1);

    // normalized RHS, zero initial guess
    RHS2->setSeed(846930886);
    RHS2->randomize();
    RHS2->norm2(norms);
    RHS2->scale(1 / norms[0]);

    X2->putScalar((SC)0.0);

    int iterations = 1;
    H->Iterate(*RHS2, *X2, iterations);

    X2->norm2(norms);
    out << "after apply, ||X2|| = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) << norms[0] << std::endl;

    RCP<MultiVector> diff = MultiVectorFactory::Build(Op->getRowMap(), 1);
    diff->putScalar(0.0);

    diff->update(1.0, *X1, -1.0, *X2, 0.0);
    diff->norm2(norms);
    TEST_EQUALITY(norms[0] < 1e-10, true);

  } else {
    out << "This test is enabled only for linAlgebra=Tpetra." << std::endl;
  }
}  // Apply

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(TpetraOperator, Getters, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include <MueLu_UseShortNames.hpp>
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  out << "version: " << MueLu::Version() << std::endl;

  using TpetraOperatorType = MueLu::TpetraOperator<SC, LO, GO, NO>;

  if (TestHelpers::Parameters::getLib() == Xpetra::UseTpetra) {
    const auto Op = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(100);

    ////////////////////////////////////////
    //////////   WITH HIERARCHY   //////////
    ////////////////////////////////////////
    {
      const auto map = Op->getRowMap();
      auto H         = rcp(new Hierarchy());
      H->setDefaultVerbLevel(Teuchos::VERB_NONE);

      auto Finest = H->GetLevel();
      Finest->setDefaultVerbLevel(Teuchos::VERB_NONE);
      Finest->Set("A", Op);

      H->Setup();
      auto tH = rcp(new TpetraOperatorType(H));

      TEST_EQUALITY(tH->GetOperator(), Teuchos::null);
      TEST_INEQUALITY(tH->GetHierarchy(), Teuchos::null);

      TEST_INEQUALITY(tH->getRangeMap(), Teuchos::null);
      TEST_INEQUALITY(tH->getDomainMap(), Teuchos::null);

      // Hardcoded false
      TEST_EQUALITY(tH->hasTransposeApply(), false);
    }

    ///////////////////////////////////////
    //////////   WITH OPERATOR   //////////
    ///////////////////////////////////////
    {
      auto tO = rcp(new TpetraOperatorType((Teuchos::RCP<Xpetra::Operator<SC, LO, GO, NO>>)(Op)));

      TEST_INEQUALITY(tO->GetOperator(), Teuchos::null);
      TEST_EQUALITY(tO->GetHierarchy(), Teuchos::null);

      TEST_INEQUALITY(tO->getRangeMap(), Teuchos::null);
      TEST_INEQUALITY(tO->getDomainMap(), Teuchos::null);

      // Hardcoded false
      TEST_EQUALITY(tO->hasTransposeApply(), false);
    }
  }
}

// Reproducer for Trilinos issue #15062: callers that only have Teuchos::RCP<const Tpetra::Operator>
// could not use CreateTpetraPreconditioner without const_cast.  The overload taking RCP<const Operator>
// must match the non-const overload (same MG iterate).
TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(TpetraOperator, CreatePreconditioner_ConstOperator, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include <MueLu_UseShortNames.hpp>
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  out << "version: " << MueLu::Version() << std::endl;

#if defined(HAVE_MUELU_IFPACK2) && defined(HAVE_MUELU_AMESOS2)
  typedef Tpetra::CrsMatrix<SC, LO, GO, NO> tpetra_crsmatrix_type;
  typedef Tpetra::Operator<SC, LO, GO, NO> tpetra_operator_type;
  typedef typename Teuchos::ScalarTraits<SC>::magnitudeType magnitude_type;

  if (TestHelpers::Parameters::getLib() == Xpetra::UseTpetra) {
    RCP<const Teuchos::Comm<int>> comm = TestHelpers::Parameters::getDefaultComm();
    RCP<Matrix> Op                     = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(6561 * comm->getSize());
    RCP<tpetra_crsmatrix_type> tpA     = Xpetra::toTpetra(Op);

    RCP<tpetra_operator_type> opNonConst(tpA);
    RCP<const tpetra_operator_type> opConst = Teuchos::rcp_implicit_cast<const tpetra_operator_type>(tpA);

    Teuchos::ParameterList mueluList;

    RCP<MueLu::TpetraOperator<SC, LO, GO, NO>> precNonConst =
        MueLu::CreateTpetraPreconditioner<SC, LO, GO, NO>(opNonConst, mueluList);
    RCP<MueLu::TpetraOperator<SC, LO, GO, NO>> precConst =
        MueLu::CreateTpetraPreconditioner<SC, LO, GO, NO>(opConst, mueluList);

    RCP<MultiVector> RHS = MultiVectorFactory::Build(Op->getRowMap(), 1);
    RHS->setSeed(846930886);
    RHS->randomize();
    Teuchos::Array<magnitude_type> norms(1);
    RHS->norm2(norms);
    RHS->scale(1 / norms[0]);

    RCP<MultiVector> Xnc = MultiVectorFactory::Build(Op->getRowMap(), 1);
    RCP<MultiVector> Xc  = MultiVectorFactory::Build(Op->getRowMap(), 1);
    Xnc->putScalar((SC)0.0);
    Xc->putScalar((SC)0.0);

    precNonConst->apply(*(Xpetra::toTpetra(RHS)), *(Xpetra::toTpetra(Xnc)));
    precConst->apply(*(Xpetra::toTpetra(RHS)), *(Xpetra::toTpetra(Xc)));

    RCP<MultiVector> diff = MultiVectorFactory::Build(Op->getRowMap(), 1);
    diff->putScalar((SC)0.0);
    diff->update(1.0, *Xnc, -1.0, *Xc, 0.0);
    diff->norm2(norms);
    out << "|| X_nonconst - X_const ||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) << norms[0] << std::endl;
    TEST_EQUALITY(norms[0] < 1e-10, true);
  } else {
    out << "This test is enabled only for linAlgebra=Tpetra." << std::endl;
  }
#else
  out << "Skipping test because some required packages are not enabled (Tpetra, Ifpack2, Amesos2)." << std::endl;
#endif
}  // CreatePreconditioner_ConstOperator

// CreateTpetraPreconditioner(inA) with no ParameterList: const vs non-const Operator overloads.
TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(TpetraOperator, CreatePreconditioner_ConstOperator_NoParameterList, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include <MueLu_UseShortNames.hpp>
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  out << "version: " << MueLu::Version() << std::endl;

#if defined(HAVE_MUELU_IFPACK2) && defined(HAVE_MUELU_AMESOS2)
  typedef Tpetra::CrsMatrix<SC, LO, GO, NO> tpetra_crsmatrix_type;
  typedef Tpetra::Operator<SC, LO, GO, NO> tpetra_operator_type;
  typedef typename Teuchos::ScalarTraits<SC>::magnitudeType magnitude_type;

  if (TestHelpers::Parameters::getLib() == Xpetra::UseTpetra) {
    RCP<const Teuchos::Comm<int>> comm = TestHelpers::Parameters::getDefaultComm();
    RCP<Matrix> Op                     = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(6561 * comm->getSize());
    RCP<tpetra_crsmatrix_type> tpA     = Xpetra::toTpetra(Op);

    RCP<tpetra_operator_type> opNonConst(tpA);
    RCP<const tpetra_operator_type> opConst = Teuchos::rcp_implicit_cast<const tpetra_operator_type>(tpA);

    RCP<MueLu::TpetraOperator<SC, LO, GO, NO>> precNonConst = MueLu::CreateTpetraPreconditioner<SC, LO, GO, NO>(opNonConst);
    RCP<MueLu::TpetraOperator<SC, LO, GO, NO>> precConst    = MueLu::CreateTpetraPreconditioner<SC, LO, GO, NO>(opConst);

    RCP<MultiVector> RHS = MultiVectorFactory::Build(Op->getRowMap(), 1);
    RHS->setSeed(846930886);
    RHS->randomize();
    Teuchos::Array<magnitude_type> norms(1);
    RHS->norm2(norms);
    RHS->scale(1 / norms[0]);

    RCP<MultiVector> Xnc = MultiVectorFactory::Build(Op->getRowMap(), 1);
    RCP<MultiVector> Xc  = MultiVectorFactory::Build(Op->getRowMap(), 1);
    Xnc->putScalar((SC)0.0);
    Xc->putScalar((SC)0.0);

    precNonConst->apply(*(Xpetra::toTpetra(RHS)), *(Xpetra::toTpetra(Xnc)));
    precConst->apply(*(Xpetra::toTpetra(RHS)), *(Xpetra::toTpetra(Xc)));

    RCP<MultiVector> diff = MultiVectorFactory::Build(Op->getRowMap(), 1);
    diff->putScalar((SC)0.0);
    diff->update(1.0, *Xnc, -1.0, *Xc, 0.0);
    diff->norm2(norms);
    out << "|| X_nonconst - X_const ||_2 (no plist) = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) << norms[0] << std::endl;
    TEST_EQUALITY(norms[0] < 1e-10, true);
  } else {
    out << "This test is enabled only for linAlgebra=Tpetra." << std::endl;
  }
#else
  out << "Skipping test because some required packages are not enabled (Tpetra, Ifpack2, Amesos2)." << std::endl;
#endif
}  // CreatePreconditioner_ConstOperator_NoParameterList

// Smaller 1D problem (fewer rows per rank) to exercise the same const / non-const overload equivalence.
TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(TpetraOperator, CreatePreconditioner_ConstOperator_SmallGrid, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include <MueLu_UseShortNames.hpp>
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  out << "version: " << MueLu::Version() << std::endl;

#if defined(HAVE_MUELU_IFPACK2) && defined(HAVE_MUELU_AMESOS2)
  typedef Tpetra::CrsMatrix<SC, LO, GO, NO> tpetra_crsmatrix_type;
  typedef Tpetra::Operator<SC, LO, GO, NO> tpetra_operator_type;
  typedef typename Teuchos::ScalarTraits<SC>::magnitudeType magnitude_type;

  if (TestHelpers::Parameters::getLib() == Xpetra::UseTpetra) {
    RCP<const Teuchos::Comm<int>> comm = TestHelpers::Parameters::getDefaultComm();
    const GO localRows                 = 243;
    RCP<Matrix> Op                     = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(localRows * comm->getSize());
    RCP<tpetra_crsmatrix_type> tpA     = Xpetra::toTpetra(Op);

    RCP<tpetra_operator_type> opNonConst(tpA);
    RCP<const tpetra_operator_type> opConst = Teuchos::rcp_implicit_cast<const tpetra_operator_type>(tpA);

    Teuchos::ParameterList mueluList;

    RCP<MueLu::TpetraOperator<SC, LO, GO, NO>> precNonConst =
        MueLu::CreateTpetraPreconditioner<SC, LO, GO, NO>(opNonConst, mueluList);
    RCP<MueLu::TpetraOperator<SC, LO, GO, NO>> precConst =
        MueLu::CreateTpetraPreconditioner<SC, LO, GO, NO>(opConst, mueluList);

    RCP<MultiVector> RHS = MultiVectorFactory::Build(Op->getRowMap(), 1);
    RHS->setSeed(123456789);
    RHS->randomize();
    Teuchos::Array<magnitude_type> norms(1);
    RHS->norm2(norms);
    RHS->scale(1 / norms[0]);

    RCP<MultiVector> Xnc = MultiVectorFactory::Build(Op->getRowMap(), 1);
    RCP<MultiVector> Xc  = MultiVectorFactory::Build(Op->getRowMap(), 1);
    Xnc->putScalar((SC)0.0);
    Xc->putScalar((SC)0.0);

    precNonConst->apply(*(Xpetra::toTpetra(RHS)), *(Xpetra::toTpetra(Xnc)));
    precConst->apply(*(Xpetra::toTpetra(RHS)), *(Xpetra::toTpetra(Xc)));

    RCP<MultiVector> diff = MultiVectorFactory::Build(Op->getRowMap(), 1);
    diff->putScalar((SC)0.0);
    diff->update(1.0, *Xnc, -1.0, *Xc, 0.0);
    diff->norm2(norms);
    out << "|| X_nonconst - X_const ||_2 (small grid) = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) << norms[0] << std::endl;
    TEST_EQUALITY(norms[0] < 1e-10, true);
  } else {
    out << "This test is enabled only for linAlgebra=Tpetra." << std::endl;
  }
#else
  out << "Skipping test because some required packages are not enabled (Tpetra, Ifpack2, Amesos2)." << std::endl;
#endif
}  // CreatePreconditioner_ConstOperator_SmallGrid

// Hierarchy depth from CreateTpetraPreconditioner must not depend on const vs non-const Operator handle.
TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(TpetraOperator, CreatePreconditioner_ConstOperator_HierarchyLevels, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include <MueLu_UseShortNames.hpp>
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  out << "version: " << MueLu::Version() << std::endl;

#if defined(HAVE_MUELU_IFPACK2) && defined(HAVE_MUELU_AMESOS2)
  typedef Tpetra::CrsMatrix<SC, LO, GO, NO> tpetra_crsmatrix_type;
  typedef Tpetra::Operator<SC, LO, GO, NO> tpetra_operator_type;

  if (TestHelpers::Parameters::getLib() == Xpetra::UseTpetra) {
    RCP<const Teuchos::Comm<int>> comm = TestHelpers::Parameters::getDefaultComm();
    RCP<Matrix> Op                     = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(2187 * comm->getSize());
    RCP<tpetra_crsmatrix_type> tpA     = Xpetra::toTpetra(Op);

    RCP<tpetra_operator_type> opNonConst(tpA);
    RCP<const tpetra_operator_type> opConst = Teuchos::rcp_implicit_cast<const tpetra_operator_type>(tpA);

    Teuchos::ParameterList mueluList;

    RCP<MueLu::TpetraOperator<SC, LO, GO, NO>> precNonConst =
        MueLu::CreateTpetraPreconditioner<SC, LO, GO, NO>(opNonConst, mueluList);
    RCP<MueLu::TpetraOperator<SC, LO, GO, NO>> precConst =
        MueLu::CreateTpetraPreconditioner<SC, LO, GO, NO>(opConst, mueluList);

    const int levelsNC = precNonConst->GetHierarchy()->GetNumLevels();
    const int levelsC  = precConst->GetHierarchy()->GetNumLevels();
    out << "numLevels non-const handle: " << levelsNC << ", const handle: " << levelsC << std::endl;
    TEST_EQUALITY(levelsNC, levelsC);
  } else {
    out << "This test is enabled only for linAlgebra=Tpetra." << std::endl;
  }
#else
  out << "Skipping test because some required packages are not enabled (Tpetra, Ifpack2, Amesos2)." << std::endl;
#endif
}  // CreatePreconditioner_ConstOperator_HierarchyLevels

// Two-column multivector: const vs non-const CreateTpetraPreconditioner apply must agree column-wise.
TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(TpetraOperator, CreatePreconditioner_ConstOperator_TwoVectors, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include <MueLu_UseShortNames.hpp>
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  out << "version: " << MueLu::Version() << std::endl;

#if defined(HAVE_MUELU_IFPACK2) && defined(HAVE_MUELU_AMESOS2)
  typedef Tpetra::CrsMatrix<SC, LO, GO, NO> tpetra_crsmatrix_type;
  typedef Tpetra::Operator<SC, LO, GO, NO> tpetra_operator_type;
  typedef typename Teuchos::ScalarTraits<SC>::magnitudeType magnitude_type;

  if (TestHelpers::Parameters::getLib() == Xpetra::UseTpetra) {
    RCP<const Teuchos::Comm<int>> comm = TestHelpers::Parameters::getDefaultComm();
    RCP<Matrix> Op                     = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(729 * comm->getSize());
    RCP<tpetra_crsmatrix_type> tpA     = Xpetra::toTpetra(Op);

    RCP<tpetra_operator_type> opNonConst(tpA);
    RCP<const tpetra_operator_type> opConst = Teuchos::rcp_implicit_cast<const tpetra_operator_type>(tpA);

    Teuchos::ParameterList mueluList;

    RCP<MueLu::TpetraOperator<SC, LO, GO, NO>> precNonConst =
        MueLu::CreateTpetraPreconditioner<SC, LO, GO, NO>(opNonConst, mueluList);
    RCP<MueLu::TpetraOperator<SC, LO, GO, NO>> precConst =
        MueLu::CreateTpetraPreconditioner<SC, LO, GO, NO>(opConst, mueluList);

    const int numVec     = 2;
    RCP<MultiVector> RHS = MultiVectorFactory::Build(Op->getRowMap(), numVec);
    RHS->setSeed(97531);
    RHS->randomize();
    Teuchos::Array<magnitude_type> colNorms(numVec);
    RHS->norm2(colNorms());
    for (int j = 0; j < numVec; ++j) {
      auto col = RHS->getVectorNonConst(j);
      col->scale(1 / colNorms[j]);
    }

    RCP<MultiVector> Xnc = MultiVectorFactory::Build(Op->getRowMap(), numVec);
    RCP<MultiVector> Xc  = MultiVectorFactory::Build(Op->getRowMap(), numVec);
    Xnc->putScalar((SC)0.0);
    Xc->putScalar((SC)0.0);

    precNonConst->apply(*(Xpetra::toTpetra(RHS)), *(Xpetra::toTpetra(Xnc)));
    precConst->apply(*(Xpetra::toTpetra(RHS)), *(Xpetra::toTpetra(Xc)));

    RCP<MultiVector> diff = MultiVectorFactory::Build(Op->getRowMap(), numVec);
    diff->putScalar((SC)0.0);
    diff->update(1.0, *Xnc, -1.0, *Xc, 0.0);
    Teuchos::Array<magnitude_type> diffNorms(numVec);
    diff->norm2(diffNorms());
    magnitude_type maxColDiff = Teuchos::ScalarTraits<magnitude_type>::zero();
    for (int j = 0; j < numVec; ++j) {
      if (diffNorms[j] > maxColDiff)
        maxColDiff = diffNorms[j];
    }
    out << "max_j || (X_nc - X_c)_j ||_2 (2 vectors) = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) << maxColDiff << std::endl;
    TEST_EQUALITY(maxColDiff < 1e-10, true);
  } else {
    out << "This test is enabled only for linAlgebra=Tpetra." << std::endl;
  }
#else
  out << "Skipping test because some required packages are not enabled (Tpetra, Ifpack2, Amesos2)." << std::endl;
#endif
}  // CreatePreconditioner_ConstOperator_TwoVectors

// ReuseTpetraPreconditioner with RCP<const CrsMatrix> vs RCP<CrsMatrix> after the same fine-matrix update.
TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(TpetraOperator, ReuseTpetraPreconditioner_ConstCrsMatrix, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include <MueLu_UseShortNames.hpp>
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  out << "version: " << MueLu::Version() << std::endl;

#if defined(HAVE_MUELU_IFPACK2) && defined(HAVE_MUELU_AMESOS2)
  typedef Tpetra::CrsMatrix<SC, LO, GO, NO> tpetra_crsmatrix_type;
  typedef Tpetra::Operator<SC, LO, GO, NO> tpetra_operator_type;
  typedef typename Teuchos::ScalarTraits<SC>::magnitudeType magnitude_type;

  if (TestHelpers::Parameters::getLib() == Xpetra::UseTpetra) {
    RCP<const Teuchos::Comm<int>> comm = TestHelpers::Parameters::getDefaultComm();
    RCP<Matrix> Op                     = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(2187 * comm->getSize());
    RCP<tpetra_crsmatrix_type> tpA     = Xpetra::toTpetra(Op);

    RCP<tpetra_operator_type> opNonConst(tpA);
    RCP<const tpetra_operator_type> opConst = Teuchos::rcp_implicit_cast<const tpetra_operator_type>(tpA);

    Teuchos::ParameterList mueluList;

    RCP<MueLu::TpetraOperator<SC, LO, GO, NO>> precNonConst =
        MueLu::CreateTpetraPreconditioner<SC, LO, GO, NO>(opNonConst, mueluList);
    RCP<MueLu::TpetraOperator<SC, LO, GO, NO>> precConst =
        MueLu::CreateTpetraPreconditioner<SC, LO, GO, NO>(opConst, mueluList);

    // Same fine matrix, same reuse entry point: non-const vs const CrsMatrix handle.
    MueLu::ReuseTpetraPreconditioner(tpA, *precNonConst);
    RCP<const tpetra_crsmatrix_type> tpAconst = Teuchos::rcp_implicit_cast<const tpetra_crsmatrix_type>(tpA);
    MueLu::ReuseTpetraPreconditioner(tpAconst, *precConst);

    RCP<MultiVector> RHS = MultiVectorFactory::Build(Op->getRowMap(), 1);
    RHS->setSeed(314159265);
    RHS->randomize();
    Teuchos::Array<magnitude_type> norms(1);
    RHS->norm2(norms);
    RHS->scale(1 / norms[0]);

    RCP<MultiVector> Xnc = MultiVectorFactory::Build(Op->getRowMap(), 1);
    RCP<MultiVector> Xc  = MultiVectorFactory::Build(Op->getRowMap(), 1);
    Xnc->putScalar((SC)0.0);
    Xc->putScalar((SC)0.0);

    precNonConst->apply(*(Xpetra::toTpetra(RHS)), *(Xpetra::toTpetra(Xnc)));
    precConst->apply(*(Xpetra::toTpetra(RHS)), *(Xpetra::toTpetra(Xc)));

    RCP<MultiVector> diff = MultiVectorFactory::Build(Op->getRowMap(), 1);
    diff->putScalar((SC)0.0);
    diff->update(1.0, *Xnc, -1.0, *Xc, 0.0);
    diff->norm2(norms);
    out << "|| X_nonconst - X_const ||_2 (after reuse) = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) << norms[0] << std::endl;
    TEST_EQUALITY(norms[0] < 1e-10, true);
  } else {
    out << "This test is enabled only for linAlgebra=Tpetra." << std::endl;
  }
#else
  out << "Skipping test because some required packages are not enabled (Tpetra, Ifpack2, Amesos2)." << std::endl;
#endif
}  // ReuseTpetraPreconditioner_ConstCrsMatrix

// Regression: Teuchos::RCP<Tpetra::CrsMatrix> used to be ambiguous between the overloads taking
// RCP<Tpetra::Operator> and RCP<const Tpetra::Operator> (e.g. Zoltan2 Sphynx).  The dedicated
// RCP<CrsMatrix> overload must compile and agree with the RCP<Operator> entry point.
TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(TpetraOperator, CreatePreconditioner_RcpCrsMatrixOverload, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include <MueLu_UseShortNames.hpp>
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  out << "version: " << MueLu::Version() << std::endl;

#if defined(HAVE_MUELU_IFPACK2) && defined(HAVE_MUELU_AMESOS2)
  typedef Tpetra::CrsMatrix<SC, LO, GO, NO> tpetra_crsmatrix_type;
  typedef Tpetra::Operator<SC, LO, GO, NO> tpetra_operator_type;
  typedef typename Teuchos::ScalarTraits<SC>::magnitudeType magnitude_type;

  if (TestHelpers::Parameters::getLib() == Xpetra::UseTpetra) {
    RCP<const Teuchos::Comm<int>> comm = TestHelpers::Parameters::getDefaultComm();
    RCP<Matrix> Op                     = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(6561 * comm->getSize());
    RCP<tpetra_crsmatrix_type> tpA     = Xpetra::toTpetra(Op);

    Teuchos::ParameterList mueluList;

    RCP<MueLu::TpetraOperator<SC, LO, GO, NO>> precFromCrs =
        MueLu::CreateTpetraPreconditioner<SC, LO, GO, NO>(tpA, mueluList);

    RCP<tpetra_operator_type> opOnly(tpA);
    RCP<MueLu::TpetraOperator<SC, LO, GO, NO>> precFromOp =
        MueLu::CreateTpetraPreconditioner<SC, LO, GO, NO>(opOnly, mueluList);

    RCP<MultiVector> RHS = MultiVectorFactory::Build(Op->getRowMap(), 1);
    RHS->setSeed(846930886);
    RHS->randomize();
    Teuchos::Array<magnitude_type> norms(1);
    RHS->norm2(norms);
    RHS->scale(1 / norms[0]);

    RCP<MultiVector> Xcrs = MultiVectorFactory::Build(Op->getRowMap(), 1);
    RCP<MultiVector> Xop  = MultiVectorFactory::Build(Op->getRowMap(), 1);
    Xcrs->putScalar((SC)0.0);
    Xop->putScalar((SC)0.0);

    precFromCrs->apply(*(Xpetra::toTpetra(RHS)), *(Xpetra::toTpetra(Xcrs)));
    precFromOp->apply(*(Xpetra::toTpetra(RHS)), *(Xpetra::toTpetra(Xop)));

    RCP<MultiVector> diff = MultiVectorFactory::Build(Op->getRowMap(), 1);
    diff->putScalar(0.0);
    diff->update(1.0, *Xcrs, -1.0, *Xop, 0.0);
    diff->norm2(norms);
    out << "|| X_rcp_crs - X_rcp_op ||_2 = " << std::setiosflags(std::ios::fixed) << std::setprecision(10) << norms[0] << std::endl;
    TEST_EQUALITY(norms[0] < 1e-10, true);
  } else {
    out << "This test is enabled only for linAlgebra=Tpetra." << std::endl;
  }
#else
  out << "Skipping test because some required packages are not enabled (Tpetra, Ifpack2, Amesos2)." << std::endl;
#endif
}  // CreatePreconditioner_RcpCrsMatrixOverload

#define MUELU_ETI_GROUP(Scalar, LocalOrdinal, GlobalOrdinal, Node)                                                                                    \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(TpetraOperator, Apply, Scalar, LocalOrdinal, GlobalOrdinal, Node)                                              \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(TpetraOperator, Getters, Scalar, LocalOrdinal, GlobalOrdinal, Node)                                            \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(TpetraOperator, CreatePreconditioner_ConstOperator, Scalar, LocalOrdinal, GlobalOrdinal, Node)                 \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(TpetraOperator, CreatePreconditioner_ConstOperator_NoParameterList, Scalar, LocalOrdinal, GlobalOrdinal, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(TpetraOperator, CreatePreconditioner_ConstOperator_SmallGrid, Scalar, LocalOrdinal, GlobalOrdinal, Node)       \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(TpetraOperator, CreatePreconditioner_ConstOperator_HierarchyLevels, Scalar, LocalOrdinal, GlobalOrdinal, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(TpetraOperator, CreatePreconditioner_ConstOperator_TwoVectors, Scalar, LocalOrdinal, GlobalOrdinal, Node)      \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(TpetraOperator, ReuseTpetraPreconditioner_ConstCrsMatrix, Scalar, LocalOrdinal, GlobalOrdinal, Node)           \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(TpetraOperator, CreatePreconditioner_RcpCrsMatrixOverload, Scalar, LocalOrdinal, GlobalOrdinal, Node)

#include <MueLu_ETI_4arg.hpp>

}  // namespace MueLuTests
