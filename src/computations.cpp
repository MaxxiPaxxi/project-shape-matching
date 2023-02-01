#include "computations.h"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <igl/readOBJ.h>
#include <iostream>

namespace deformations {

bool mesh::load(std::string const& filepath)
{
  this->reset();
  bool const result = igl::readOBJ(filepath, V0_, F_);
  x_ = V0_;
  v_.resizeLike(x_);
  f_.resizeLike(x_);

  fixed_.resize(x_.rows(), 1);
  fixed_.setConstant(false);

  v_.setZero(); // Set velocity to zero
  f_.setZero(); // Set forces to zero

  Q_ = V0_.rowwise() - V0_.colwise().mean();
  assert((Q_.rows() == V0_.rows() && Q_.cols() == V0_.cols(), "This failed"));

  double mi = 1.0;

  AqqInv_.setZero();

  for (int i = 0; i < Q_.rows(); i++)
    {
      assert((Q_.cols() == 3, "Q_ must have right dimension"));
      Eigen::Vector3d qi = Q_.block(i,0, 1, Q_.cols()).transpose();
      assert((qi.rows() == 3 && qi.cols() == 1, "qi must have right dimension"));
      AqqInv_ += mi * qi * qi.transpose();
    }
  assert((AqqInv_.determinant() != 0.0, "Aqq must be invertible"));
  AqqInv_ = AqqInv_.inverse().eval();

  AqqInvQuadratic_.setZero();

  Qquadratic_.resize(Q_.rows(), 9);
  Qquadratic_.setZero();
  for (int i = 0; i < Qquadratic_.rows(); i++)
    {
      Eigen::RowVector3d qi = Q_.block(i, 0, 1, Q_.cols());
      Eigen::Matrix<double, 9, 1> qitilde;
      qitilde << qi.x(), qi.y(), qi.z(), qi.x() * qi.x(), qi.y() * qi.y(), qi.z() * qi.z(), qi.x() * qi.y(), qi.y() * qi.z(), qi.z() * qi.x();
      Qquadratic_.block(i, 0, 1, 9) = qitilde.transpose();
      AqqInvQuadratic_ += mi * qitilde * qitilde.transpose();
    }

  Eigen::SelfAdjointEigenSolver<decltype(AqqInvQuadratic_)> solver(AqqInvQuadratic_);
  std::ostringstream oss{};
  //auto const info = solver.info();

  Eigen::Matrix<double, 9,9> inversion;
  inversion.setZero();

  Eigen::Matrix<double, 9, 1> eigenvalues = solver.eigenvalues();
  if (std::abs(solver.eigenvalues().minCoeff()) < 1e-6)
  {
    eigenvalues.array() += perturbation_;
  }

  for (int i = 0; i < 9; i++)
    {
      inversion(i,i) = 1.0 / eigenvalues(i);
    }
  AqqInvQuadratic_ = solver.eigenvectors() * inversion * solver.eigenvectors().transpose();
  return result;
}

Eigen::MatrixX3d mesh::apply_force(Eigen::Vector3d location, Eigen::Vector3d force)
{
  Eigen::Vector3d direction = force.normalized();
  Eigen::VectorXd t(x_.rows());
  t.setZero();

  for (int i = 0; i < t.rows(); i++)
    {
      Eigen::Vector3d r = x_.block(i, 0, 1, x_.cols()).transpose() - location;
      double s1 = r.dot(-direction);
      double theta = std::acos(s1 / (r.norm() * direction.norm()));
      double s2 = r.norm() * std::sin(theta);

      t(i) = std::abs(theta) < M_PI / 2.0 ? std::exp(-theta) : 0.0;
    }
  double tmax = t.maxCoeff();
  double force_interpolation_weight = 1.0 / tmax;

  Eigen::MatrixXd f(t.rows(), force.rows());
  for (int i = 0; i < force.rows(); i++)
    {
      assert(((t.array() * force_interpolation_weight).maxCoeff() <= 1.0 && (t.array() * force_interpolation_weight).minCoeff() >= 0.0, "Test function must be in between 0 and 1"));
      f.block(0, i, f.rows(), 1) = t.array() * force_interpolation_weight * force(i);
    }
  f_ += f;

  return f;
}

void mesh::integrate(double dt)
{
      if (type_ != deformation_type::linear)
    {
        return;
    }

    // pi = xi - xcm
    Eigen::Vector3d xcm = x_.colwise().mean().transpose();
    Eigen::MatrixX3d P  = x_.rowwise() - xcm.transpose();

    Eigen::Matrix3d Apq;
    Apq.setZero();

    assert((P.rows() == Q_.rows() && P.cols() == Q_.cols(), "P and Q need to have the right dimension"));

    // Apq = [sum(mi*pi*qit)]
    for (int i = 0; i < P.rows(); i++)
    {
      assert((P.cols() == 3, "P.cols() must be of dimension 3"));
      Eigen::Vector3d qi = Q_.block(i, 0, 1, Q_.cols()).transpose();
      Eigen::Vector3d pi = P.block(i, 0, 1, P.cols()).transpose();
      assert((pi.rows() == 3 && pi.cols() == 1, "pi must have right dimension"));

        // assume each vertex mass is 1.0 for the time being
        double mi = 1.0;
        Apq += mi * pi * qi.transpose();
    }

    Eigen::Matrix3d ApqSquared = (Apq.transpose() * Apq);

    Eigen::SelfAdjointEigenSolver<decltype(ApqSquared)> eigensolver(ApqSquared);
    Eigen::Matrix3d SInv = eigensolver.operatorInverseSqrt();

    Eigen::Matrix3d R = Apq * SInv;

    Eigen::Matrix3d A   = Apq * AqqInv_;
    double volume = A.determinant();
    A                   = A / std::cbrt(volume);

    double beta       = beta_;
    Eigen::Matrix3d T = beta_ * A + (1 - beta_) * R;

    Eigen::MatrixX3d g = (T * Q_.transpose()).transpose().rowwise() + xcm.transpose();

    double alpha = dt / tau_;
    double mi    = 1.0;

    Eigen::MatrixX3d elasticity   = alpha * (g - x_);
    Eigen::MatrixX3d acceleration = f_ / mi;

    for (int i = 0; i < x_.rows(); i++)
    {
        if (fixed_(i))
        {
            v_.block(i, 0, 1, 3) = Eigen::RowVector3d{0.0, 0.0, 0.0};
        }
        else
        {
            v_.block(i, 0, 1, 3) = v_.block(i, 0, 1, 3) + (elasticity.block(i, 0, 1, 3) / dt) +
                                   (dt * acceleration.block(i, 0, 1, 3)) -
                                   Rb_ * v_.block(i, 0, 1, 3);
        }

        x_.block(i, 0, 1, 3) = x_.block(i, 0, 1, 3) + dt * v_.block(i, 0, 1, 3);
    }

    f_.setZero();
}

void mesh::integrate_quadratic(double dt)
{
    if (type_ != deformation_type::quadratic)
    {
        return;
    }

    double mi = 1.0;

    Eigen::RowVector3d xcm = x_.colwise().mean();
    Eigen::MatrixX3d P     = x_.rowwise() - xcm;

    // compute quadratic terms
    Eigen::Matrix<double, 3, 9> ApqTilde;
    ApqTilde.setZero();

    assert((Qquadratic_.rows() == Q_.rows(), "Qquadratic must have right dimension"));
    assert((Qquadratic_.rows() == x_.rows(), "Qquadratic must have right dimension"));
    assert((Qquadratic_.rows() == V0_.rows(), "Qquadratic must have right dimension"));
    for (int i = 0; i < Qquadratic_.rows(); i++)
    {
        Eigen::Vector3d pi = P.block(i, 0, 1, 3).transpose();
        Eigen::Matrix<double, 1, 9> qitilde_transpose = Qquadratic_.block(i, 0, 1, 9);

        ApqTilde += mi * pi * qitilde_transpose;
    }

    // compute linear terms
    Eigen::Matrix3d Apq;
    Apq.setZero();

    assert((P.rows() == Q_.rows() && P.cols() == Q_.cols(), "P must have right dimension"));

    // Apq = [sum(mi*pi*qit)]
    for (int i = 0; i < P.rows(); i++)
    {
        assert((P.cols() == 3, "P.cols must be of dimension 3"));
        Eigen::Vector3d qi = Q_.block(i, 0, 1, Q_.cols()).transpose();
        Eigen::Vector3d pi = P.block(i, 0, 1, P.cols()).transpose();
        assert((pi.rows() == 3 && pi.cols() == 1, "pi must have right dimension"));

        // assume each vertex mass is 1.0 for the time being
        double mi = 1.0;
        Apq += mi * pi * qi.transpose();
    }

    Eigen::Matrix3d ApqSquared = (Apq.transpose() * Apq);

    Eigen::SelfAdjointEigenSolver<decltype(ApqSquared)> eigensolver(ApqSquared);
    Eigen::Matrix3d SInv = eigensolver.operatorInverseSqrt();

    Eigen::Matrix3d R = Apq * SInv;

    Eigen::Matrix<double, 3, 9> Rtilde;
    Rtilde.setZero();
    Rtilde.block(0, 0, 3, 3) = R;

    Eigen::Matrix<double, 3, 9> Atilde = ApqTilde * AqqInvQuadratic_;

    // ensure volume preservation
    // auto A = Atilde.block(0, 0, 3, 3);
    // double const volume = A.determinant();
    // A = A / std::cbrt(volume);

    Eigen::Matrix<double, 3, 9> T = beta_ * Atilde + (1 - beta_) * Rtilde;

    Eigen::MatrixX3d shape = (T * Qquadratic_.transpose()).transpose();
    Eigen::MatrixX3d g     = shape.rowwise() + xcm;

    double alpha = dt / tau_;

    Eigen::MatrixX3d elasticity   = alpha * (g - x_);
    Eigen::MatrixX3d acceleration = f_ / mi;

    for (int i = 0; i < x_.rows(); i++)
    {
        if (fixed_(i))
        {
            v_.block(i, 0, 1, 3) = Eigen::RowVector3d{0.0, 0.0, 0.0};
        }
        else
        {
            v_.block(i, 0, 1, 3) = v_.block(i, 0, 1, 3) + (elasticity.block(i, 0, 1, 3) / dt) +
                                   (dt * acceleration.block(i, 0, 1, 3)) -
                                   Rb_ * v_.block(i, 0, 1, 3);

            Eigen::RowVector3d vel = v_.block(i, 0, 1, 3);

            vel + vel;
        }

        x_.block(i, 0, 1, 3) = x_.block(i, 0, 1, 3) + dt * v_.block(i, 0, 1, 3);
    }

    f_.setZero();
}

void mesh::reset()
{
    V0_.setZero();
    Q_.setZero();
    AqqInv_.setZero();
    Qquadratic_.setZero();
    AqqInvQuadratic_.setZero();
    x_.setZero();
    v_.setZero();
    fixed_.setZero();
    f_.setZero();
    F_.setZero();
}
} // Namespace deformations;
