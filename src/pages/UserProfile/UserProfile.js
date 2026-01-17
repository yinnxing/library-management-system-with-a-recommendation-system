import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useUser } from '../../contexts/UserContext';
import UserApi from '../../api/UserApi';
import styles from './UserProfile.module.css';
import '../../styles/design-system.css';
import { message } from 'antd';

const UserProfile = () => {
  const { user } = useUser();
  const navigate = useNavigate();
  const [isEditing, setIsEditing] = useState(false);
  const [profileData, setProfileData] = useState({
    username: '',
    email: '',
    dob: '',
    gender: '',
    createdAt: '',
    roles: []
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [passwordData, setPasswordData] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: ''
  });
  const [isChangingPassword, setIsChangingPassword] = useState(false);

  useEffect(() => {
    // Check if user is logged in
    if (!user) {
      navigate('/login');
      return;
    }

    // Fetch user profile data using security context
    const fetchUserProfile = async () => {
      try {
        setLoading(true);
        setError('');
        const response = await UserApi.getMyInfo();
        const userData = response.data.result;
        
        setProfileData({
          username: userData.username || '',
          email: userData.email || '',
          dob: userData.dob || '',
          gender: userData.gender || '',
          createdAt: userData.createdAt || '',
          roles: userData.roles || []
        });
      } catch (err) {
        console.error('Error fetching user profile:', err);
        setError('Failed to load profile data');
        message.error('Failed to load profile data');
      } finally {
        setLoading(false);
      }
    };

    fetchUserProfile();
  }, [user, navigate]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setProfileData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handlePasswordChange = (e) => {
    const { name, value } = e.target;
    setPasswordData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleEditToggle = () => {
    if (isEditing) {
      // Reset form if canceling edit
      const resetForm = async () => {
        try {
          const response = await UserApi.getMyInfo();
          const userData = response.data.result;
          setProfileData({
            username: userData.username || '',
            email: userData.email || '',
            dob: userData.dob || '',
            gender: userData.gender || '',
            createdAt: userData.createdAt || '',
            roles: userData.roles || []
          });
        } catch (err) {
          console.error('Error refreshing user profile:', err);
          message.error('Không thể làm mới dữ liệu hồ sơ');
        }
      };
      resetForm();
    }
    
    setIsEditing(prev => !prev);
  };

  const handlePasswordToggle = () => {
    setIsChangingPassword(prev => !prev);
    if (isChangingPassword) {
      // Reset password form
      setPasswordData({
        currentPassword: '',
        newPassword: '',
        confirmPassword: ''
      });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // Use the existing updateUserProfile method with userId
      await UserApi.updateUserProfile({
        username: profileData.username,
        dob: profileData.dob,
        gender: profileData.gender
      });
      message.success('Profile updated successfully');
      setIsEditing(false);
    } catch (err) {
      console.error('Error updating profile:', err);
      setError('Failed to update profile');
      message.error('Failed to update profile');
    }
  };

  const handlePasswordSubmit = async (e) => {
    e.preventDefault();
    
    if (passwordData.newPassword !== passwordData.confirmPassword) {
      setError('New passwords do not match');
      message.error('New passwords do not match');
      return;
    }
    
    try {
      await UserApi.changePassword(passwordData.currentPassword, passwordData.newPassword, passwordData.confirmPassword);
      message.success('Password changed successfully');
      setIsChangingPassword(false);
      setPasswordData({
        currentPassword: '',
        newPassword: '',
        confirmPassword: ''
      });
    } catch (err) {
      console.error('Error changing password:', err);
      setError('Failed to change password');
      message.error('Failed to change password');
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleDateString('vi-VN', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  const formatDobForInput = (dateString) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toISOString().split('T')[0]; // Format: YYYY-MM-DD
  };

  const getGenderDisplay = (gender) => {
    switch (gender?.toLowerCase()) {
      case 'male':
        return 'Nam';
      case 'female':
        return 'Nữ';
      case 'other':
        return 'Khác';
      default:
        return 'Chưa cập nhật';
    }
  };

  const handleEditButtonClick = (e) => {
    e.preventDefault();
    e.stopPropagation();
    handleEditToggle();
  };

  if (loading) {
    return (
      <div className={styles.loadingContainer}>
        <div className={styles.loadingSpinner}></div>
        <p className={styles.loadingText}>Đang tải thông tin hồ sơ...</p>
      </div>
    );
  }

  return (
    <div className={styles.profileContainer}>
      <div className={styles.profileHeader}>
        <h1 className={styles.pageTitle}>
          <span className={styles.titleIcon}>👤</span>
          Hồ Sơ Người Dùng
        </h1>
        <p className={styles.pageSubtitle}>
          Quản lý thông tin cá nhân và cài đặt tài khoản của bạn
        </p>
      </div>

      {error && (
        <div className={styles.errorMessage}>
          <span className={styles.errorIcon}>⚠️</span>
          {error}
        </div>
      )}
      
      <div className={styles.profileCard}>
        <div className={styles.profileInfo}>
          <div className={styles.avatarSection}>
            <div className={styles.avatarContainer}>
              <img 
                src="https://media.istockphoto.com/id/1443209389/vi/vec-to/s%C3%A1ch-v%E1%BA%BD-ngu%E1%BB%87ch-ngo%E1%BA%A1c.jpg?s=612x612&w=0&k=20&c=a-bJw6Ic1FLS5Ri4TlRqOh0bREGyVO_2W52seuNupuk=" 
                alt="Profile" 
                className={styles.avatar} 
              />
            </div>
            <div className={styles.userInfo}>
              <h2 className={styles.userName}>{profileData.username || profileData.email}</h2>
              <p className={styles.userEmail}>{profileData.email}</p>
              <div className={styles.userMeta}>
                <span className={styles.joinDate}>
                  Tham gia: {formatDate(profileData.createdAt)}
                </span>
                {profileData.dob && (
                  <span className={styles.userDob}>
                    Sinh nhật: {formatDate(profileData.dob)}
                  </span>
                )}
                {profileData.gender && (
                  <span className={styles.userGender}>
                    Giới tính: {getGenderDisplay(profileData.gender)}
                  </span>
                )}
                {profileData.roles && profileData.roles.length > 0 && (
                  <span className={styles.userRole}>
                    {profileData.roles.map(role => role.name).join(', ')}
                  </span>
                )}
              </div>
            </div>
          </div>

          <form onSubmit={handleSubmit} className={styles.profileForm}>
            <div className={styles.formGroup}>
              <label className={styles.formLabel}>Tên người dùng</label>
              <input 
                type="text" 
                name="username" 
                value={profileData.username} 
                onChange={handleInputChange}
                disabled={!isEditing}
                className={styles.formInput}
                placeholder="Nhập tên người dùng"
              />
            </div>
            
            <div className={styles.formGroup}>
              <label className={styles.formLabel}>Email</label>
              <input 
                type="email" 
                name="email" 
                value={profileData.email} 
                disabled={true}
                className={`${styles.formInput} ${styles.disabled}`}
                title="Email không thể thay đổi"
              />
            </div>
            
            <div className={styles.formGroup}>
              <label className={styles.formLabel}>Ngày sinh</label>
              <input 
                type="date" 
                name="dob" 
                value={formatDobForInput(profileData.dob)} 
                onChange={handleInputChange}
                disabled={!isEditing}
                className={styles.formInput}
                max={new Date().toISOString().split('T')[0]} // Prevent future dates
              />
            </div>
            
            <div className={styles.formGroup}>
              <label className={styles.formLabel}>Giới tính</label>
              <select 
                name="gender" 
                value={profileData.gender} 
                onChange={handleInputChange}
                disabled={!isEditing}
                className={styles.formSelect}
              >
                <option value="">Chọn giới tính</option>
                <option value="male">Nam</option>
                <option value="female">Nữ</option>
                <option value="other">Khác</option>
              </select>
            </div>
            
            <div className={styles.buttonGroup}>
              {isEditing ? (
                <>
                  <button type="submit" className={styles.saveButton}>
                    <span className={styles.buttonIcon}>💾</span>
                    Lưu Thay Đổi
                  </button>
                  <button type="button" onClick={handleEditToggle} className={styles.cancelButton}>
                    <span className={styles.buttonIcon}>❌</span>
                    Hủy
                  </button>
                </>
              ) : (
                <button type="button" onClick={handleEditButtonClick} className={styles.editButton}>
                  <span className={styles.buttonIcon}>✏️</span>
                  Chỉnh Sửa Hồ Sơ
                </button>
              )}
            </div>
          </form>
        </div>
        
        <div className={styles.passwordSection}>
          <h3 className={styles.sectionTitle}>
            <span className={styles.sectionIcon}>🔒</span>
            Đổi Mật Khẩu
          </h3>
          {isChangingPassword ? (
            <form onSubmit={handlePasswordSubmit} className={styles.passwordForm}>
              <div className={styles.formGroup}>
                <label className={styles.formLabel}>Mật khẩu hiện tại</label>
                <input 
                  type="password" 
                  name="currentPassword" 
                  value={passwordData.currentPassword} 
                  onChange={handlePasswordChange}
                  required
                  className={styles.formInput}
                  placeholder="Nhập mật khẩu hiện tại"
                />
              </div>
              
              <div className={styles.formGroup}>
                <label className={styles.formLabel}>Mật khẩu mới</label>
                <input 
                  type="password" 
                  name="newPassword" 
                  value={passwordData.newPassword} 
                  onChange={handlePasswordChange}
                  required
                  className={styles.formInput}
                  placeholder="Nhập mật khẩu mới"
                />
              </div>
              
              <div className={styles.formGroup}>
                <label className={styles.formLabel}>Xác nhận mật khẩu mới</label>
                <input 
                  type="password" 
                  name="confirmPassword" 
                  value={passwordData.confirmPassword} 
                  onChange={handlePasswordChange}
                  required
                  className={styles.formInput}
                  placeholder="Nhập lại mật khẩu mới"
                />
              </div>
              
              <div className={styles.buttonGroup}>
                <button type="submit" className={styles.saveButton}>
                  <span className={styles.buttonIcon}>🔐</span>
                  Đổi Mật Khẩu
                </button>
                <button type="button" onClick={handlePasswordToggle} className={styles.cancelButton}>
                  <span className={styles.buttonIcon}>❌</span>
                  Hủy
                </button>
              </div>
            </form>
          ) : (
            <button onClick={handlePasswordToggle} className={styles.changePasswordButton}>
              <span className={styles.buttonIcon}>🔑</span>
              Đổi Mật Khẩu
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default UserProfile;
