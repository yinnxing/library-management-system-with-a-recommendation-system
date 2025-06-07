package com.library.booksystem.dto.request;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.Size;
import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.FieldDefaults;

import java.time.LocalDate;

@Data
@NoArgsConstructor
@AllArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE)
public class UpdateProfileRequest {
    @Size(min = 4, message = "USERNAME_INVALID")
    String username;
    
    @Email(message = "EMAIL_INVALID")
    String email;
    
    LocalDate dob;
    
    String gender;
} 