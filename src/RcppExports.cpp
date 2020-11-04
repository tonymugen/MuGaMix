// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// testLpostNR
Rcpp::List testLpostNR(const std::vector<double>& yVec, const int32_t& d, const int32_t& Npop, std::vector<double>& theta, const std::vector<double>& P, const int32_t& ind, const double& limit, const double& incr);
RcppExport SEXP _MuGaMix_testLpostNR(SEXP yVecSEXP, SEXP dSEXP, SEXP NpopSEXP, SEXP thetaSEXP, SEXP PSEXP, SEXP indSEXP, SEXP limitSEXP, SEXP incrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<double>& >::type yVec(yVecSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type d(dSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Npop(NpopSEXP);
    Rcpp::traits::input_parameter< std::vector<double>& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type P(PSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type ind(indSEXP);
    Rcpp::traits::input_parameter< const double& >::type limit(limitSEXP);
    Rcpp::traits::input_parameter< const double& >::type incr(incrSEXP);
    rcpp_result_gen = Rcpp::wrap(testLpostNR(yVec, d, Npop, theta, P, ind, limit, incr));
    return rcpp_result_gen;
END_RCPP
}
// testLpostP
Rcpp::List testLpostP(const std::vector<double>& yVec, const int32_t& d, const int32_t& Npop, const std::vector<double>& theta, std::vector<double>& Phi, const int32_t& ind, const double& limit, const double& incr);
RcppExport SEXP _MuGaMix_testLpostP(SEXP yVecSEXP, SEXP dSEXP, SEXP NpopSEXP, SEXP thetaSEXP, SEXP PhiSEXP, SEXP indSEXP, SEXP limitSEXP, SEXP incrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<double>& >::type yVec(yVecSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type d(dSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Npop(NpopSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< std::vector<double>& >::type Phi(PhiSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type ind(indSEXP);
    Rcpp::traits::input_parameter< const double& >::type limit(limitSEXP);
    Rcpp::traits::input_parameter< const double& >::type incr(incrSEXP);
    rcpp_result_gen = Rcpp::wrap(testLpostP(yVec, d, Npop, theta, Phi, ind, limit, incr));
    return rcpp_result_gen;
END_RCPP
}
// testLpostLocNR
Rcpp::List testLpostLocNR(const std::vector<double>& yVec, const int32_t& d, const int32_t& Npop, std::vector<double>& theta, const std::vector<double>& iSigTheta, const int32_t& ind, const double& limit, const double& incr);
RcppExport SEXP _MuGaMix_testLpostLocNR(SEXP yVecSEXP, SEXP dSEXP, SEXP NpopSEXP, SEXP thetaSEXP, SEXP iSigThetaSEXP, SEXP indSEXP, SEXP limitSEXP, SEXP incrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<double>& >::type yVec(yVecSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type d(dSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Npop(NpopSEXP);
    Rcpp::traits::input_parameter< std::vector<double>& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type iSigTheta(iSigThetaSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type ind(indSEXP);
    Rcpp::traits::input_parameter< const double& >::type limit(limitSEXP);
    Rcpp::traits::input_parameter< const double& >::type incr(incrSEXP);
    rcpp_result_gen = Rcpp::wrap(testLpostLocNR(yVec, d, Npop, theta, iSigTheta, ind, limit, incr));
    return rcpp_result_gen;
END_RCPP
}
// testGradNR
Rcpp::List testGradNR(const std::vector<double>& yVec, const int32_t& d, const int32_t& Npop, std::vector<double>& theta, const std::vector<double>& P, const int32_t& ind, const double& limit, const double& incr);
RcppExport SEXP _MuGaMix_testGradNR(SEXP yVecSEXP, SEXP dSEXP, SEXP NpopSEXP, SEXP thetaSEXP, SEXP PSEXP, SEXP indSEXP, SEXP limitSEXP, SEXP incrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<double>& >::type yVec(yVecSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type d(dSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Npop(NpopSEXP);
    Rcpp::traits::input_parameter< std::vector<double>& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type P(PSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type ind(indSEXP);
    Rcpp::traits::input_parameter< const double& >::type limit(limitSEXP);
    Rcpp::traits::input_parameter< const double& >::type incr(incrSEXP);
    rcpp_result_gen = Rcpp::wrap(testGradNR(yVec, d, Npop, theta, P, ind, limit, incr));
    return rcpp_result_gen;
END_RCPP
}
// testGradP
Rcpp::List testGradP(const std::vector<double>& yVec, const int32_t& d, const int32_t& Npop, const std::vector<double>& theta, std::vector<double>& Phi, const int32_t& ind, const double& limit, const double& incr);
RcppExport SEXP _MuGaMix_testGradP(SEXP yVecSEXP, SEXP dSEXP, SEXP NpopSEXP, SEXP thetaSEXP, SEXP PhiSEXP, SEXP indSEXP, SEXP limitSEXP, SEXP incrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<double>& >::type yVec(yVecSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type d(dSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Npop(NpopSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< std::vector<double>& >::type Phi(PhiSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type ind(indSEXP);
    Rcpp::traits::input_parameter< const double& >::type limit(limitSEXP);
    Rcpp::traits::input_parameter< const double& >::type incr(incrSEXP);
    rcpp_result_gen = Rcpp::wrap(testGradP(yVec, d, Npop, theta, Phi, ind, limit, incr));
    return rcpp_result_gen;
END_RCPP
}
// testGradLocNR
Rcpp::List testGradLocNR(const std::vector<double>& yVec, const int32_t& d, const int32_t& Npop, std::vector<double>& theta, const std::vector<double>& iSigTheta, const int32_t& ind, const double& limit, const double& incr);
RcppExport SEXP _MuGaMix_testGradLocNR(SEXP yVecSEXP, SEXP dSEXP, SEXP NpopSEXP, SEXP thetaSEXP, SEXP iSigThetaSEXP, SEXP indSEXP, SEXP limitSEXP, SEXP incrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<double>& >::type yVec(yVecSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type d(dSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Npop(NpopSEXP);
    Rcpp::traits::input_parameter< std::vector<double>& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type iSigTheta(iSigThetaSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type ind(indSEXP);
    Rcpp::traits::input_parameter< const double& >::type limit(limitSEXP);
    Rcpp::traits::input_parameter< const double& >::type incr(incrSEXP);
    rcpp_result_gen = Rcpp::wrap(testGradLocNR(yVec, d, Npop, theta, iSigTheta, ind, limit, incr));
    return rcpp_result_gen;
END_RCPP
}
// testLpostSigNR
Rcpp::List testLpostSigNR(const std::vector<double>& yVec, const int32_t& d, const int32_t& Npop, const std::vector<double>& theta, std::vector<double>& iSigTheta, const int32_t& ind, const double& limit, const double& incr);
RcppExport SEXP _MuGaMix_testLpostSigNR(SEXP yVecSEXP, SEXP dSEXP, SEXP NpopSEXP, SEXP thetaSEXP, SEXP iSigThetaSEXP, SEXP indSEXP, SEXP limitSEXP, SEXP incrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<double>& >::type yVec(yVecSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type d(dSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Npop(NpopSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< std::vector<double>& >::type iSigTheta(iSigThetaSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type ind(indSEXP);
    Rcpp::traits::input_parameter< const double& >::type limit(limitSEXP);
    Rcpp::traits::input_parameter< const double& >::type incr(incrSEXP);
    rcpp_result_gen = Rcpp::wrap(testLpostSigNR(yVec, d, Npop, theta, iSigTheta, ind, limit, incr));
    return rcpp_result_gen;
END_RCPP
}
// testLpostLoc
double testLpostLoc(const std::vector<double>& yVec, const std::vector<int32_t>& lnFac, const int32_t& Npop, const std::vector<double>& theta, const std::vector<double>& iSigTheta);
RcppExport SEXP _MuGaMix_testLpostLoc(SEXP yVecSEXP, SEXP lnFacSEXP, SEXP NpopSEXP, SEXP thetaSEXP, SEXP iSigThetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<double>& >::type yVec(yVecSEXP);
    Rcpp::traits::input_parameter< const std::vector<int32_t>& >::type lnFac(lnFacSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Npop(NpopSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type iSigTheta(iSigThetaSEXP);
    rcpp_result_gen = Rcpp::wrap(testLpostLoc(yVec, lnFac, Npop, theta, iSigTheta));
    return rcpp_result_gen;
END_RCPP
}
// lpTestLI
Rcpp::List lpTestLI(const std::vector<double>& yVec, const std::vector<int32_t>& lnFac, const int32_t& Npop, std::vector<double>& theta, const std::vector<double>& iSigTheta, const int32_t& ind, const double& limit, const double& incr);
RcppExport SEXP _MuGaMix_lpTestLI(SEXP yVecSEXP, SEXP lnFacSEXP, SEXP NpopSEXP, SEXP thetaSEXP, SEXP iSigThetaSEXP, SEXP indSEXP, SEXP limitSEXP, SEXP incrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<double>& >::type yVec(yVecSEXP);
    Rcpp::traits::input_parameter< const std::vector<int32_t>& >::type lnFac(lnFacSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Npop(NpopSEXP);
    Rcpp::traits::input_parameter< std::vector<double>& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type iSigTheta(iSigThetaSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type ind(indSEXP);
    Rcpp::traits::input_parameter< const double& >::type limit(limitSEXP);
    Rcpp::traits::input_parameter< const double& >::type incr(incrSEXP);
    rcpp_result_gen = Rcpp::wrap(lpTestLI(yVec, lnFac, Npop, theta, iSigTheta, ind, limit, incr));
    return rcpp_result_gen;
END_RCPP
}
// gradTestLI
Rcpp::List gradTestLI(const std::vector<double>& yVec, const std::vector<int32_t>& lnFac, const int32_t& Npop, std::vector<double>& theta, const std::vector<double>& iSigTheta, const int32_t& ind, const double& limit, const double& incr);
RcppExport SEXP _MuGaMix_gradTestLI(SEXP yVecSEXP, SEXP lnFacSEXP, SEXP NpopSEXP, SEXP thetaSEXP, SEXP iSigThetaSEXP, SEXP indSEXP, SEXP limitSEXP, SEXP incrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<double>& >::type yVec(yVecSEXP);
    Rcpp::traits::input_parameter< const std::vector<int32_t>& >::type lnFac(lnFacSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Npop(NpopSEXP);
    Rcpp::traits::input_parameter< std::vector<double>& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type iSigTheta(iSigThetaSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type ind(indSEXP);
    Rcpp::traits::input_parameter< const double& >::type limit(limitSEXP);
    Rcpp::traits::input_parameter< const double& >::type incr(incrSEXP);
    rcpp_result_gen = Rcpp::wrap(gradTestLI(yVec, lnFac, Npop, theta, iSigTheta, ind, limit, incr));
    return rcpp_result_gen;
END_RCPP
}
// lpTestSI
Rcpp::List lpTestSI(const std::vector<double>& yVec, const std::vector<int32_t>& lnFac, const int32_t& Npop, const std::vector<double>& theta, std::vector<double>& iSigTheta, const int32_t& ind, const double& limit, const double& incr);
RcppExport SEXP _MuGaMix_lpTestSI(SEXP yVecSEXP, SEXP lnFacSEXP, SEXP NpopSEXP, SEXP thetaSEXP, SEXP iSigThetaSEXP, SEXP indSEXP, SEXP limitSEXP, SEXP incrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<double>& >::type yVec(yVecSEXP);
    Rcpp::traits::input_parameter< const std::vector<int32_t>& >::type lnFac(lnFacSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Npop(NpopSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< std::vector<double>& >::type iSigTheta(iSigThetaSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type ind(indSEXP);
    Rcpp::traits::input_parameter< const double& >::type limit(limitSEXP);
    Rcpp::traits::input_parameter< const double& >::type incr(incrSEXP);
    rcpp_result_gen = Rcpp::wrap(lpTestSI(yVec, lnFac, Npop, theta, iSigTheta, ind, limit, incr));
    return rcpp_result_gen;
END_RCPP
}
// gradTestSInr
Rcpp::List gradTestSInr(const std::vector<double>& yVec, const int32_t& d, const int32_t& Npop, const std::vector<double>& theta, std::vector<double>& iSigTheta, const int32_t& ind, const double& limit, const double& incr);
RcppExport SEXP _MuGaMix_gradTestSInr(SEXP yVecSEXP, SEXP dSEXP, SEXP NpopSEXP, SEXP thetaSEXP, SEXP iSigThetaSEXP, SEXP indSEXP, SEXP limitSEXP, SEXP incrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<double>& >::type yVec(yVecSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type d(dSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Npop(NpopSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< std::vector<double>& >::type iSigTheta(iSigThetaSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type ind(indSEXP);
    Rcpp::traits::input_parameter< const double& >::type limit(limitSEXP);
    Rcpp::traits::input_parameter< const double& >::type incr(incrSEXP);
    rcpp_result_gen = Rcpp::wrap(gradTestSInr(yVec, d, Npop, theta, iSigTheta, ind, limit, incr));
    return rcpp_result_gen;
END_RCPP
}
// gradTestSI
Rcpp::List gradTestSI(const std::vector<double>& yVec, const std::vector<int32_t>& lnFac, const int32_t& Npop, const std::vector<double>& theta, std::vector<double>& iSigTheta, const int32_t& ind, const double& limit, const double& incr);
RcppExport SEXP _MuGaMix_gradTestSI(SEXP yVecSEXP, SEXP lnFacSEXP, SEXP NpopSEXP, SEXP thetaSEXP, SEXP iSigThetaSEXP, SEXP indSEXP, SEXP limitSEXP, SEXP incrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<double>& >::type yVec(yVecSEXP);
    Rcpp::traits::input_parameter< const std::vector<int32_t>& >::type lnFac(lnFacSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Npop(NpopSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< std::vector<double>& >::type iSigTheta(iSigThetaSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type ind(indSEXP);
    Rcpp::traits::input_parameter< const double& >::type limit(limitSEXP);
    Rcpp::traits::input_parameter< const double& >::type incr(incrSEXP);
    rcpp_result_gen = Rcpp::wrap(gradTestSI(yVec, lnFac, Npop, theta, iSigTheta, ind, limit, incr));
    return rcpp_result_gen;
END_RCPP
}
// vbFit
Rcpp::List vbFit(const std::vector<double>& yVec, const int32_t& d, const int32_t& nGroups, const double& alphaPr, const double& sigSqPr, const double& covRatio, const int32_t nReps);
RcppExport SEXP _MuGaMix_vbFit(SEXP yVecSEXP, SEXP dSEXP, SEXP nGroupsSEXP, SEXP alphaPrSEXP, SEXP sigSqPrSEXP, SEXP covRatioSEXP, SEXP nRepsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<double>& >::type yVec(yVecSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type d(dSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type nGroups(nGroupsSEXP);
    Rcpp::traits::input_parameter< const double& >::type alphaPr(alphaPrSEXP);
    Rcpp::traits::input_parameter< const double& >::type sigSqPr(sigSqPrSEXP);
    Rcpp::traits::input_parameter< const double& >::type covRatio(covRatioSEXP);
    Rcpp::traits::input_parameter< const int32_t >::type nReps(nRepsSEXP);
    rcpp_result_gen = Rcpp::wrap(vbFit(yVec, d, nGroups, alphaPr, sigSqPr, covRatio, nReps));
    return rcpp_result_gen;
END_RCPP
}
// vbFitMiss
Rcpp::List vbFitMiss(std::vector<double>& yVec, const int32_t& d, const int32_t& nGroups, const double& alphaPr, const double& sigSqPr, const double& covRatio, const int32_t nReps);
RcppExport SEXP _MuGaMix_vbFitMiss(SEXP yVecSEXP, SEXP dSEXP, SEXP nGroupsSEXP, SEXP alphaPrSEXP, SEXP sigSqPrSEXP, SEXP covRatioSEXP, SEXP nRepsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::vector<double>& >::type yVec(yVecSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type d(dSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type nGroups(nGroupsSEXP);
    Rcpp::traits::input_parameter< const double& >::type alphaPr(alphaPrSEXP);
    Rcpp::traits::input_parameter< const double& >::type sigSqPr(sigSqPrSEXP);
    Rcpp::traits::input_parameter< const double& >::type covRatio(covRatioSEXP);
    Rcpp::traits::input_parameter< const int32_t >::type nReps(nRepsSEXP);
    rcpp_result_gen = Rcpp::wrap(vbFitMiss(yVec, d, nGroups, alphaPr, sigSqPr, covRatio, nReps));
    return rcpp_result_gen;
END_RCPP
}
// runSamplerNR
Rcpp::List runSamplerNR(const std::vector<double>& yVec, const int32_t& d, const int32_t& Npop, const int32_t& Nadapt, const int32_t& Nsamp, const int32_t& Nthin, const int32_t& Nchains);
RcppExport SEXP _MuGaMix_runSamplerNR(SEXP yVecSEXP, SEXP dSEXP, SEXP NpopSEXP, SEXP NadaptSEXP, SEXP NsampSEXP, SEXP NthinSEXP, SEXP NchainsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<double>& >::type yVec(yVecSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type d(dSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Npop(NpopSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Nadapt(NadaptSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Nsamp(NsampSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Nthin(NthinSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Nchains(NchainsSEXP);
    rcpp_result_gen = Rcpp::wrap(runSamplerNR(yVec, d, Npop, Nadapt, Nsamp, Nthin, Nchains));
    return rcpp_result_gen;
END_RCPP
}
// runSampler
Rcpp::List runSampler(const std::vector<double>& yVec, const std::vector<int32_t>& lnFac, const int32_t& Npop, const int32_t& Nadapt, const int32_t& Nsamp, const int32_t& Nthin, const int32_t& Nchains);
RcppExport SEXP _MuGaMix_runSampler(SEXP yVecSEXP, SEXP lnFacSEXP, SEXP NpopSEXP, SEXP NadaptSEXP, SEXP NsampSEXP, SEXP NthinSEXP, SEXP NchainsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<double>& >::type yVec(yVecSEXP);
    Rcpp::traits::input_parameter< const std::vector<int32_t>& >::type lnFac(lnFacSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Npop(NpopSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Nadapt(NadaptSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Nsamp(NsampSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Nthin(NthinSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Nchains(NchainsSEXP);
    rcpp_result_gen = Rcpp::wrap(runSampler(yVec, lnFac, Npop, Nadapt, Nsamp, Nthin, Nchains));
    return rcpp_result_gen;
END_RCPP
}
// runSamplerMiss
Rcpp::List runSamplerMiss(const std::vector<double>& yVec, const std::vector<int32_t>& lnFac, const std::vector<int32_t>& missIDs, const int32_t& Npop, const int32_t& Nadapt, const int32_t& Nsamp, const int32_t& Nthin, const int32_t& Nchains);
RcppExport SEXP _MuGaMix_runSamplerMiss(SEXP yVecSEXP, SEXP lnFacSEXP, SEXP missIDsSEXP, SEXP NpopSEXP, SEXP NadaptSEXP, SEXP NsampSEXP, SEXP NthinSEXP, SEXP NchainsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<double>& >::type yVec(yVecSEXP);
    Rcpp::traits::input_parameter< const std::vector<int32_t>& >::type lnFac(lnFacSEXP);
    Rcpp::traits::input_parameter< const std::vector<int32_t>& >::type missIDs(missIDsSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Npop(NpopSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Nadapt(NadaptSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Nsamp(NsampSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Nthin(NthinSEXP);
    Rcpp::traits::input_parameter< const int32_t& >::type Nchains(NchainsSEXP);
    rcpp_result_gen = Rcpp::wrap(runSamplerMiss(yVec, lnFac, missIDs, Npop, Nadapt, Nsamp, Nthin, Nchains));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_MuGaMix_testLpostNR", (DL_FUNC) &_MuGaMix_testLpostNR, 8},
    {"_MuGaMix_testLpostP", (DL_FUNC) &_MuGaMix_testLpostP, 8},
    {"_MuGaMix_testLpostLocNR", (DL_FUNC) &_MuGaMix_testLpostLocNR, 8},
    {"_MuGaMix_testGradNR", (DL_FUNC) &_MuGaMix_testGradNR, 8},
    {"_MuGaMix_testGradP", (DL_FUNC) &_MuGaMix_testGradP, 8},
    {"_MuGaMix_testGradLocNR", (DL_FUNC) &_MuGaMix_testGradLocNR, 8},
    {"_MuGaMix_testLpostSigNR", (DL_FUNC) &_MuGaMix_testLpostSigNR, 8},
    {"_MuGaMix_testLpostLoc", (DL_FUNC) &_MuGaMix_testLpostLoc, 5},
    {"_MuGaMix_lpTestLI", (DL_FUNC) &_MuGaMix_lpTestLI, 8},
    {"_MuGaMix_gradTestLI", (DL_FUNC) &_MuGaMix_gradTestLI, 8},
    {"_MuGaMix_lpTestSI", (DL_FUNC) &_MuGaMix_lpTestSI, 8},
    {"_MuGaMix_gradTestSInr", (DL_FUNC) &_MuGaMix_gradTestSInr, 8},
    {"_MuGaMix_gradTestSI", (DL_FUNC) &_MuGaMix_gradTestSI, 8},
    {"_MuGaMix_vbFit", (DL_FUNC) &_MuGaMix_vbFit, 7},
    {"_MuGaMix_vbFitMiss", (DL_FUNC) &_MuGaMix_vbFitMiss, 7},
    {"_MuGaMix_runSamplerNR", (DL_FUNC) &_MuGaMix_runSamplerNR, 7},
    {"_MuGaMix_runSampler", (DL_FUNC) &_MuGaMix_runSampler, 7},
    {"_MuGaMix_runSamplerMiss", (DL_FUNC) &_MuGaMix_runSamplerMiss, 8},
    {NULL, NULL, 0}
};

RcppExport void R_init_MuGaMix(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
