# User Profile Management APIs

This document describes the new APIs for user profile management and password changes.

## API Endpoints

### 1. Update User Profile
**Endpoint:** `PUT /users/profile`

**Description:** Allows authenticated users to update their profile information including username, email, date of birth, and gender.

**Request Headers:**
- `Authorization: Bearer <access_token>`
- `Content-Type: application/json`

**Request Body:**
```json
{
  "username": "newusername",
  "email": "newemail@example.com",
  "dob": "1990-01-15",
  "gender": "Male"
}
```

**Request Body Fields:**
- `username` (optional): New username (minimum 4 characters)
- `email` (optional): New email address (must be valid email format)
- `dob` (optional): Date of birth in YYYY-MM-DD format
- `gender` (optional): Gender information

**Response:**
```json
{
  "code": 1000,
  "message": "Success",
  "result": {
    "userId": "user-uuid",
    "username": "newusername",
    "email": "newemail@example.com",
    "role": "USER",
    "dob": "1990-01-15",
    "gender": "Male"
  }
}
```

**Error Responses:**
- `1001`: User existed (username or email already taken)
- `1002`: Username invalid (less than 4 characters)
- `1016`: Email format is invalid
- `1005`: User not exist
- `1007`: Unauthenticated

### 2. Change Password
**Endpoint:** `PUT /users/change-password`

**Description:** Allows authenticated users to change their password by providing current password and new password.

**Request Headers:**
- `Authorization: Bearer <access_token>`
- `Content-Type: application/json`

**Request Body:**
```json
{
  "currentPassword": "oldpassword123",
  "newPassword": "newpassword123",
  "confirmPassword": "newpassword123"
}
```

**Request Body Fields:**
- `currentPassword` (required): Current password for verification
- `newPassword` (required): New password (minimum 8 characters)
- `confirmPassword` (required): Confirmation of new password (must match newPassword)

**Response:**
```json
{
  "code": 1000,
  "message": "Password changed successfully"
}
```

**Error Responses:**
- `1017`: Current password is required
- `1018`: New password is required
- `1019`: Confirm password is required
- `1003`: Password invalid (less than 8 characters)
- `1015`: New password and confirm password do not match
- `1007`: Unauthenticated (current password is incorrect)
- `1005`: User not exist

## Validation Rules

### Username
- Minimum 4 characters
- Must be unique across all users
- Cannot be null if provided

### Email
- Must be valid email format
- Must be unique across all users
- Cannot be null if provided

### Password
- Minimum 8 characters
- Current password must be correct for password changes
- New password and confirm password must match

### Date of Birth
- Must be in valid date format (YYYY-MM-DD)
- Optional field

### Gender
- Free text field
- Optional field

## Security Features

1. **Authentication Required**: All endpoints require valid JWT token
2. **Current User Only**: Users can only update their own profile
3. **Password Verification**: Current password must be provided and verified before changing
4. **Unique Constraints**: Username and email uniqueness is enforced
5. **Input Validation**: All inputs are validated according to business rules

## Usage Examples

### Update Profile with cURL
```bash
curl -X PUT http://localhost:8080/users/profile \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "newusername",
    "email": "newemail@example.com",
    "dob": "1990-01-15",
    "gender": "Male"
  }'
```

### Change Password with cURL
```bash
curl -X PUT http://localhost:8080/users/change-password \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Content-Type: application/json" \
  -d '{
    "currentPassword": "oldpassword123",
    "newPassword": "newpassword123",
    "confirmPassword": "newpassword123"
  }'
```

## Notes

- All fields in the update profile request are optional. Only provided fields will be updated.
- The system uses MapStruct for mapping between DTOs and entities with null value property mapping strategy set to IGNORE.
- Password encoding is handled using Spring Security's PasswordEncoder.
- The APIs follow the existing application's response format with ApiResponse wrapper. 